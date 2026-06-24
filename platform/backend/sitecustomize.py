"""Backend Python startup customizations.

On some Windows installations, Python 3.12's ``platform.machine()`` can block
inside a WMI query. SQLAlchemy calls that function during import, so the backend
can hang before FastAPI starts. Use the processor architecture environment
value instead for backend processes.
"""

from __future__ import annotations

import os
import platform
import sys
from collections import namedtuple


def _windows_machine() -> str:
    return (
        os.environ.get("PYTHON_PLATFORM_MACHINE")
        or os.environ.get("PROCESSOR_ARCHITEW6432")
        or os.environ.get("PROCESSOR_ARCHITECTURE")
        or "AMD64"
    )


def _windows_system() -> str:
    return "Windows"


def _windows_release() -> str:
    version_info = sys.getwindowsversion()
    if version_info.major == 10 and version_info.build >= 22000:
        return "11"
    return str(version_info.major)


def _windows_version() -> str:
    version_info = sys.getwindowsversion()
    return f"{version_info.major}.{version_info.minor}.{version_info.build}"


def _windows_processor() -> str:
    return os.environ.get("PROCESSOR_IDENTIFIER") or _windows_machine()


_UnameResult = namedtuple("uname_result", "system node release version machine processor")


def _windows_uname() -> tuple[str, str, str, str, str, str]:
    return _UnameResult(
        system=_windows_system(),
        node=os.environ.get("COMPUTERNAME", ""),
        release=_windows_release(),
        version=_windows_version(),
        machine=_windows_machine(),
        processor=_windows_processor(),
    )


def _windows_win32_ver() -> tuple[str, str, str, str]:
    return (_windows_release(), _windows_version(), "", "")


if sys.platform.startswith("win"):
    platform.system = _windows_system
    platform.release = _windows_release
    platform.version = _windows_version
    platform.machine = _windows_machine
    platform.processor = _windows_processor
    platform.uname = _windows_uname
    platform.win32_ver = _windows_win32_ver
