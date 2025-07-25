#
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# distutils: language = c++


import sys


cdef void _log_callback(int lvl, const char * msg) with gil:
    """Callback function to redirect logs to `sys.stdout`.

    Parameters
    ----------
    lvl : int
        Level of the logging message as defined by rapids-logger.
    msg : char *
        Message to be logged
    """
    print(msg.decode('utf-8'), end='')


cdef void _log_flush() with gil:
    """Callback function to flush logs."""
    if sys.stdout is not None:
        sys.stdout.flush()


def _verbose_to_level(verbose: bool | int) -> level_enum:
    """Parse the common `verbose` parameter into a `level_enum`."""
    if verbose is True:
        return level_enum.debug
    elif verbose is False:
        return level_enum.info
    else:
        return level_enum(6 - verbose)


def _verbose_from_level(level: level_enum) -> int:
    """Convert a `level_enum` back into an equivalent `verbose` parameter value."""
    return 6 - int(level)


cdef class set_level:
    """Set the logging level for the current thread.

    Can optionally be used as a contextmanager to reset the log level back to
    its previous value after the context exits.

    Parameters
    ----------
    level : str or level_enum
        The logging level to set.

    Examples
    --------
    >>> from cuml.internals import logger

    Set the log level in this thread to `info`.

    >>> logger.set_level("info")  # doctest: +SKIP

    Set the log level in this thread to `debug` within a single context.

    >>> with logger.set_level("debug"):  # doctest: +SKIP
    ...     debug("A debug message")
    """
    cdef level_enum _prev_level

    def __cinit__(self, level):
        if not isinstance(level, level_enum):
            level = level_enum[level]
        self._prev_level = default_logger().level()
        default_logger().set_level(level)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        default_logger().set_level(self._prev_level)


def get_level() -> level_enum:
    """Get the current logging level."""
    return default_logger().level()


def should_log_for(level_enum level):
    """Check if messages at the given logging level will be logged or not.

    This is a useful check to avoid doing unnecessary logging work.

    Parameters
    ----------
    level : level_enum
        Logging level to be set.

    Returns
    -------
    bool
        Whether the given logging level will be logged or not.

    Examples
    --------
    >>> if should_log_for(level_enum.info):  # doctest: +SKIP
    ...     # avoid expensive operation if it won't be logged
    ...     my_message = construct_expensive_message()
    ...     info(my_message)
    """
    return default_logger().should_log(level)


cdef _log(level_enum lvl, str msg):
    """
    Internal function to log a message at a given level.

    Parameters
    ----------
    lvl : level_enum
        Logging level to be set.
    msg : str
        Message to be logged.
    """
    cdef string s = msg.encode("UTF-8")
    default_logger().log(lvl, s)


def trace(msg):
    """Logs a message at trace level.

    Examples
    --------
    >>> info("This is a trace message")  # doctest: +SKIP

    Parameters
    ----------
    msg : str
        Message to be logged.
    """
    # No trace level in Python so we use the closest thing, debug.
    _log(level_enum.trace, msg)


def debug(msg):
    """Logs a message at debug level.

    Examples
    --------
    >>> info("This is a debug message")  # doctest: +SKIP

    Parameters
    ----------
    msg : str
        Message to be logged.
    """
    _log(level_enum.debug, msg)


def info(msg):
    """Logs a message at info level.

    Examples
    --------
    >>> info("This is an info message")  # doctest: +SKIP

    Parameters
    ----------
    msg : str
        Message to be logged.
    """
    _log(level_enum.info, msg)


def warn(msg):
    """Logs a message at warning level.

    Examples
    --------
    >>> warn("This is a warning message")  # doctest: +SKIP

    Parameters
    ----------
    msg : str
        Message to be logged.
    """
    _log(level_enum.warn, msg)


def error(msg):
    """Logs a message at error level.

    Examples
    --------
    >>> error("This is an error message")  # doctest: +SKIP

    Parameters
    ----------
    msg : str
        Message to be logged.
    """
    _log(level_enum.error, msg)


def critical(msg):
    """Logs a message at critical level.

    Examples
    --------
    >>> critical("This is a critical message")  # doctest: +SKIP

    Parameters
    ----------
    msg : str
        Message to be logged.
    """
    _log(level_enum.critical, msg)


def flush():
    """
    Flush the logs.
    """
    default_logger().flush()


# Change the default sink to use a callback to redirect to sys.stdout
set_default_sink(<sink_ptr> make_shared[callback_sink_mt](_log_callback, _log_flush))
# Update the existing local logger to use the new default sink
default_logger().sinks().clear()
default_logger().sinks().push_back(default_sink())
