"""Keep Windows from idle-sleeping while a benchmark runner is alive.

Uses SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED), the same
request media players make. It changes no power settings and lapses
automatically when the calling process exits. No-op on other platforms.
"""
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def stay_awake():
    if sys.platform != "win32":
        return False
    import ctypes
    return bool(ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))


def release():
    if sys.platform != "win32":
        return
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
