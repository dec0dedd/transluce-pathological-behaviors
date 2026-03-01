from __future__ import annotations

import typing
from typing import Awaitable, Callable, ParamSpec, TypeVar

import anyio
import anyio.abc

Args = typing.TypeVarTuple("Args")
T = typing.TypeVar("T")
U = typing.TypeVar("U")


class FutureFinishedWithExceptionError(Exception):
    """An error raised when a future finished with an exception."""


class FutureNotSetError(Exception):
    """An error raised when a future is not set."""


class SimpleFuture[T]:
    def __init__(self):
        self.event = anyio.Event()
        self.result = None
        self.exception = None

    def set_result(self, result: T):
        assert not self.event.is_set(), "Result already set"
        self.result = result
        self.event.set()

    def set_from_task(
        self,
        task_fn: Callable[[*Args], typing.Awaitable[T]],
        set_exception: bool = True,
        catch_exception: bool = False,
    ) -> Callable[[*Args], typing.Awaitable[None]]:
        if catch_exception and not set_exception:
            raise ValueError("set_exception must be True if catch_exception is True")

        async def wrapped_task_fn(*args: *Args):
            try:
                result = await task_fn(*args)
            except Exception as e:
                if set_exception:
                    self.set_exception(e)
                    if not catch_exception:
                        raise
                else:
                    raise
            else:
                self.set_result(result)

        return wrapped_task_fn

    def set_exception(self, exception: Exception):
        assert not self.event.is_set(), "Result already set"
        self.exception = exception
        self.event.set()

    def has_result(self) -> bool:
        return self.event.is_set()

    async def wait_for_result(self) -> T:
        await self.event.wait()
        if self.exception:
            raise FutureFinishedWithExceptionError(
                f"Future finished with an exception (of type {type(self.exception).__name__})!"
            ) from self.exception
        assert self.result is not None
        return self.result

    def get(self) -> T:
        if not self.has_result():
            raise FutureNotSetError("Result not set")
        if self.exception:
            raise self.exception
        assert self.result is not None
        return self.result


def future_from_start_soon[
    T, *Args
](
    task_group: anyio.abc.TaskGroup,
    task_fn: Callable[[*Args], Awaitable[T]],
    *args: *Args,
    catch_exception: bool = False,
) -> SimpleFuture[T]:
    result = SimpleFuture[T]()
    task_group.start_soon(
        result.set_from_task(task_fn, set_exception=True, catch_exception=catch_exception),
        *args,
    )
    return result


P = ParamSpec("P")
T = TypeVar("T")
