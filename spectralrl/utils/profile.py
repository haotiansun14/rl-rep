import atexit
import functools
import os
import time
import traceback
from typing import Any, Callable, Optional

import torch


class profile(object):
    """
    Decorator to profile the function and to print elapsed time at the exit of the program, can
    also function as a context manager to time a piece of wrapped codes within `with` statement.

    Notes
    -----
    Usage of this module:
    1. decorating a named function by `@profile(identifier=xxx)`
    2. wrap the codes to time by `with` statement `with profile(identifier=xxx) as p:`

    Parameters
    ----------
    identifier :  The unique tag for disguishing the profile from others. If set to None, in decorator mode
        the identifier will be set to the name of the decotated function, while in context-manager mode it
        will be set to the calling position.
    activate :  Whether the profile is activated or not. Default is True.
    """
    global_registry = {}

    def __init__(self, identifier: Optional[str] = None, activate: bool = True) -> None:
        self.identifier = identifier
        self.activate = activate
        self.stash = 0   # only used for context manager

        def exit_logger():
            from UtilsRL.logger import (
                logger,  # the input is moved here to avoid circular import
            )
            elapsed_time, called_time = profile.global_registry[self.identifier]
            logger.info(
                f"[profile]: Executed {self.identifier} for {called_time} times, total elapsed time {elapsed_time}(s), avg {elapsed_time/called_time if called_time else 0}(s)."
            )
        self.exit_logger_fn = exit_logger

    def __call__(self, func: Callable) -> Callable:
        if self.activate:
            if self.identifier is None:
                self.identifier = func.__name__
            if self.identifier not in profile.global_registry:
                profile.global_registry[self.identifier] = [0, 0]
                atexit.register(self.exit_logger_fn)

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                torch.cuda.synchronize()
                beg = time.time()
                ret = func(*args, **kwargs)
                torch.cuda.synchronize()
                profile.global_registry[self.identifier][0] += time.time() - beg
                profile.global_registry[self.identifier][1] += 1
                return ret

            return wrapper
        else:
            return lambda fn: fn

    def __enter__(self):
        if self.activate:
            if self.identifier is None:
                stack = traceback.extract_stack()[0]
                cwd = os.getcwd()
                filename = stack.filename
                lineno = stack.lineno
                if filename.startswith(cwd):
                    filename = filename.replace(cwd, ".")
                self.identifier = "{}: line {}".format(filename, lineno)
            if self.identifier not in profile.global_registry:
                profile.global_registry[self.identifier] = [0, 0]
                atexit.register(self.exit_logger_fn)
            torch.cuda.synchronize()
            self.beg = time.time()
        return self

    def __exit__(self, exc_type, exc_value: Any, traceback: Any) -> None:
        if self.activate:
            torch.cuda.synchronize()
            profile.global_registry[self.identifier][0] += time.time() - self.beg + self.stash
            profile.global_registry[self.identifier][1] += 1

    def pause(self):
        torch.cuda.synchronize()
        self.stash += time.time() - self.beg
        self.beg = None

    def resume(self):
        torch.cuda.synchronize()
        self.beg = time.time()
