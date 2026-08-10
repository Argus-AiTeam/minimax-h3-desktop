#!/usr/bin/env python3
"""Project-local fallback for ``python3 -m pytest`` in the Argus dry-run path.

The execution host for this mission does not provide the external ``pytest``
package, and the current bounded task must stay local: no package downloads,
no sudo, and no global environment changes.  This module first delegates to a
real pytest installation if one exists outside the repository.  If none is
available, it runs a tiny compatibility subset sufficient for this repository's
pre-gate verifier tests: test-file discovery, zero-argument test functions,
and ``pytest.raises``.

It is intentionally small and should not be treated as a general replacement
for pytest in post-gate development environments.
"""
from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import inspect
import io
import os
import re
import shutil
import sys
import tempfile
import traceback
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Iterable

__version__ = "argus-local-fallback"


class Skipped(unittest.SkipTest):
    """Internal skip signal used by the project-local fallback runner."""


def skip(reason: str = "") -> None:
    """Skip the current fallback test without counting it as passed."""

    raise Skipped(reason or "skipped")


class RaisesContext:
    """Minimal context manager compatible with ``pytest.raises`` usage here."""

    def __init__(self, expected_exception: type[BaseException] | tuple[type[BaseException], ...], match: str | None = None):
        self.expected_exception = expected_exception
        self.match = match
        self.value: BaseException | None = None

    def __enter__(self) -> "RaisesContext":
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> bool:
        if exc_type is None or exc is None:
            raise AssertionError(f"DID NOT RAISE {self.expected_exception!r}")
        if not issubclass(exc_type, self.expected_exception):
            return False
        if self.match is not None and re.search(self.match, str(exc)) is None:
            raise AssertionError(f"exception message {str(exc)!r} does not match {self.match!r}")
        self.value = exc
        return True


def raises(expected_exception: type[BaseException] | tuple[type[BaseException], ...], *args: Any, match: str | None = None, **kwargs: Any) -> RaisesContext:
    """Return a context manager or call a function expecting an exception.

    Supports the two common forms::

        with pytest.raises(Error): ...
        pytest.raises(Error, func, *args, **kwargs)
    """

    if args and callable(args[0]):
        func = args[0]
        rest = args[1:]
        with RaisesContext(expected_exception, match=match):
            func(*rest, **kwargs)
        return RaisesContext(expected_exception, match=match)
    return RaisesContext(expected_exception, match=match)


def fail(reason: str = "") -> None:
    raise AssertionError(reason or "pytest.fail() called")


class _NoOpMark:
    def __getattr__(self, name: str) -> Callable[..., Any]:
        if name == "parametrize":
            def parametrize(argnames: str | Iterable[str], argvalues: Iterable[Any], **_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
                if isinstance(argnames, str):
                    names = [part.strip() for part in argnames.split(",") if part.strip()]
                else:
                    names = [str(part) for part in argnames]
                values = list(argvalues)

                def wrap(func: Callable[..., Any]) -> Callable[..., Any]:
                    existing = list(getattr(func, "_argus_parametrize", []))
                    existing.append((names, values))
                    setattr(func, "_argus_parametrize", existing)
                    return func

                return wrap

            return parametrize

        if name == "skip":
            def skip_marker(*args: Any, reason: str = "", **_kwargs: Any) -> Any:
                def wrap(func: Callable[..., Any]) -> Callable[..., Any]:
                    setattr(func, "_argus_skip_reason", reason or "pytest.mark.skip")
                    return func

                if args and len(args) == 1 and callable(args[0]):
                    return wrap(args[0])
                return wrap

            return skip_marker

        if name == "skipif":
            def skipif_marker(condition: Any, *args: Any, reason: str = "", **_kwargs: Any) -> Any:
                def wrap(func: Callable[..., Any]) -> Callable[..., Any]:
                    if bool(condition):
                        setattr(func, "_argus_skip_reason", reason or "pytest.mark.skipif condition is true")
                    return func

                if args and len(args) == 1 and callable(args[0]):
                    return wrap(args[0])
                return wrap

            return skipif_marker

        def decorator(*args: Any, **kwargs: Any) -> Any:
            if args and len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]

            def wrap(func: Callable[..., Any]) -> Callable[..., Any]:
                return func

            return wrap

        return decorator


mark = _NoOpMark()


def fixture(func: Callable[..., Any] | None = None, **_kwargs: Any) -> Callable[..., Any]:
    def wrap(inner: Callable[..., Any]) -> Callable[..., Any]:
        return inner

    if func is not None:
        return wrap(func)
    return wrap


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _path_entry_resolves_to_repo(entry: str) -> bool:
    try:
        return Path(entry or os.getcwd()).resolve() == _repo_root()
    except OSError:
        return False


def _delegate_to_real_pytest(argv: list[str]) -> int | None:
    """Run real pytest when it is installed outside this repository."""

    filtered_path = [entry for entry in sys.path if not _path_entry_resolves_to_repo(entry)]
    spec = importlib.machinery.PathFinder.find_spec("pytest", filtered_path)
    if spec is None or spec.origin is None:
        return None
    if Path(spec.origin).resolve() == Path(__file__).resolve():
        return None

    old_path = sys.path[:]
    previous = sys.modules.pop("pytest", None)
    try:
        sys.path = filtered_path
        real_pytest = importlib.import_module("pytest")
        real_main = getattr(real_pytest, "main", None)
        if real_main is None:
            return None
        return int(real_main(argv))
    finally:
        sys.path = old_path
        if previous is not None and "pytest" not in sys.modules:
            sys.modules["pytest"] = previous


def _iter_test_files(paths: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.rglob("test_*.py")))
        elif path.is_file() and path.name.startswith("test_") and path.suffix == ".py":
            files.append(path)
    return files


def _load_module(path: Path, index: int) -> ModuleType:
    module_name = f"_argus_fallback_pytest_{index}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load test module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _iter_test_functions(module: ModuleType) -> list[tuple[str, Callable[..., Any]]]:
    tests: list[tuple[str, Callable[..., Any]]] = []
    for name, obj in vars(module).items():
        if name.startswith("test_") and callable(obj):
            tests.append((name, obj))
    return sorted(tests, key=lambda item: item[0])


class MonkeyPatch:
    def __init__(self) -> None:
        self._env: list[tuple[str, str | None]] = []
        self._attrs: list[tuple[Any, str, Any, bool]] = []

    def setenv(self, name: str, value: str) -> None:
        self._env.append((name, os.environ.get(name)))
        os.environ[name] = value

    def delenv(self, name: str, raising: bool = True) -> None:
        if name not in os.environ:
            if raising:
                raise KeyError(name)
            self._env.append((name, None))
            return
        self._env.append((name, os.environ.get(name)))
        del os.environ[name]

    def setattr(self, target: Any, name: str, value: Any, raising: bool = True) -> None:
        had = hasattr(target, name)
        if not had and raising:
            raise AttributeError(name)
        old = getattr(target, name, None)
        self._attrs.append((target, name, old, had))
        setattr(target, name, value)

    def undo(self) -> None:
        for target, name, old, had in reversed(self._attrs):
            if had:
                setattr(target, name, old)
            else:
                try:
                    delattr(target, name)
                except AttributeError:
                    pass
        self._attrs.clear()
        for name, old in reversed(self._env):
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old
        self._env.clear()


class CaptureFixture:
    def __init__(self) -> None:
        self.out = io.StringIO()
        self.err = io.StringIO()

    def readouterr(self) -> SimpleNamespace:
        payload = SimpleNamespace(out=self.out.getvalue(), err=self.err.getvalue())
        self.out.seek(0)
        self.out.truncate(0)
        self.err.seek(0)
        self.err.truncate(0)
        return payload


def _parametrize_cases(func: Callable[..., Any]) -> list[tuple[str, dict[str, Any]]]:
    parametrize = list(getattr(func, "_argus_parametrize", []))
    if not parametrize:
        return [("", {})]
    cases: list[tuple[str, dict[str, Any]]] = [("", {})]
    for names, values in parametrize:
        expanded: list[tuple[str, dict[str, Any]]] = []
        for prefix, base_kwargs in cases:
            for index, raw in enumerate(values):
                if len(names) == 1:
                    raw_values = (raw,)
                else:
                    raw_values = tuple(raw)
                if len(raw_values) != len(names):
                    raise TypeError("parametrize value count does not match argnames")
                kwargs = dict(base_kwargs)
                kwargs.update(dict(zip(names, raw_values)))
                suffix = "-".join(str(value) for value in raw_values)
                expanded.append((f"{prefix}[{index}:{suffix}]", kwargs))
        cases = expanded
    return cases


def _build_fixture_kwargs(required: list[inspect.Parameter], param_kwargs: dict[str, Any]) -> tuple[dict[str, Any], Callable[[], None], CaptureFixture | None]:
    kwargs: dict[str, Any] = dict(param_kwargs)
    cleanups: list[Callable[[], None]] = []
    capsys: CaptureFixture | None = None
    for param in required:
        name = param.name
        if name in kwargs:
            continue
        if name == "tmp_path":
            tmp = Path(tempfile.mkdtemp(prefix="argus-pytest-"))
            kwargs[name] = tmp
            cleanups.append(lambda tmp=tmp: shutil.rmtree(tmp, ignore_errors=True))
        elif name == "monkeypatch":
            mp = MonkeyPatch()
            kwargs[name] = mp
            cleanups.append(mp.undo)
        elif name == "capsys":
            capsys = CaptureFixture()
            kwargs[name] = capsys
        else:
            raise TypeError(f"fallback pytest does not support fixture injection for {name!r}")

    def cleanup() -> None:
        for item in reversed(cleanups):
            item()

    return kwargs, cleanup, capsys


def _fallback_main(argv: list[str]) -> int:
    paths: list[str] = []
    unsupported: list[str] = []
    skip_next = False
    for idx, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg in {"-q", "-v", "-s", "--disable-warnings"} or arg.startswith("--tb"):
            continue
        if arg == "-k":
            skip_next = True
            continue
        if arg.startswith("-"):
            unsupported.append(arg)
        else:
            paths.append(arg)

    if unsupported:
        print(f"fallback pytest does not support arguments: {' '.join(unsupported)}", file=sys.stderr)
        return 2

    if not paths:
        paths = ["tests"]

    test_files = _iter_test_files(paths)
    if not test_files:
        print("fallback pytest: no tests collected", file=sys.stderr)
        return 5

    passed = 0
    skipped = 0
    failed = 0
    collected = 0
    for file_index, path in enumerate(test_files):
        try:
            module = _load_module(path, file_index)
            tests = _iter_test_functions(module)
        except BaseException:
            failed += 1
            print(f"{path}::IMPORT FAILED", file=sys.stderr)
            traceback.print_exc()
            continue

        for name, func in tests:
            for case_suffix, param_kwargs in _parametrize_cases(func):
                collected += 1
                nodeid = f"{path.as_posix()}::{name}{case_suffix}"
                cleanup: Callable[[], None] = lambda: None
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                skip_reason = getattr(func, "_argus_skip_reason", None)
                if skip_reason is not None:
                    skipped += 1
                    print(f"{nodeid} SKIPPED ({skip_reason})")
                    continue
                try:
                    signature = inspect.signature(func)
                    required = [
                        param
                        for param in signature.parameters.values()
                        if param.default is inspect.Parameter.empty
                        and param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                    ]
                    kwargs, cleanup, capsys = _build_fixture_kwargs(required, param_kwargs)
                    if capsys is not None:
                        sys.stdout = capsys.out
                        sys.stderr = capsys.err
                    func(**kwargs)
                except unittest.SkipTest as exc:
                    if sys.stdout is not old_stdout:
                        sys.stdout = old_stdout
                    if sys.stderr is not old_stderr:
                        sys.stderr = old_stderr
                    skipped += 1
                    print(f"{nodeid} SKIPPED ({exc})")
                except BaseException:
                    if sys.stdout is not old_stdout:
                        sys.stdout = old_stdout
                    if sys.stderr is not old_stderr:
                        sys.stderr = old_stderr
                    failed += 1
                    print(f"{nodeid} FAILED", file=sys.stderr)
                    traceback.print_exc()
                else:
                    if sys.stdout is not old_stdout:
                        sys.stdout = old_stdout
                    if sys.stderr is not old_stderr:
                        sys.stderr = old_stderr
                    passed += 1
                    print(f"{nodeid} PASSED")
                finally:
                    try:
                        cleanup()
                    finally:
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr

    print(f"{collected} collected, {passed} passed, {skipped} skipped, {failed} failed")
    return 0 if failed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    delegated = _delegate_to_real_pytest(argv)
    if delegated is not None:
        return delegated
    return _fallback_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
