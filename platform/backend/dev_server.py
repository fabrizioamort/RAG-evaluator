"""Development server entrypoint with cross-platform process cleanup."""

from __future__ import annotations

import argparse
import os
import re
import signal
import socket
import subprocess
import sys
import time

APP_PATH = "app.main:app"
HOST = "0.0.0.0"
PORT = 8000
PORT_RELEASE_TIMEOUT_SECONDS = 5.0


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def windows_listening_pids(port: int) -> list[int]:
    result = subprocess.run(
        ["netstat", "-ano", "-p", "tcp"],
        capture_output=True,
        text=True,
        check=False,
    )
    pids: set[int] = set()
    port_suffix = f":{port}"

    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        if parts[0] != "TCP":
            continue
        if not parts[1].endswith(port_suffix):
            continue
        if parts[3] != "LISTENING":
            continue
        try:
            pids.add(int(parts[4]))
        except ValueError:
            continue

    return sorted(pids)


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None


def posix_listening_pids(port: int) -> list[int]:
    ss_result = _run_command(["ss", "-ltnp", f"sport = :{port}"])
    if ss_result is not None and ss_result.returncode == 0:
        return sorted({int(pid) for pid in re.findall(r"pid=(\d+)", ss_result.stdout)})

    lsof_result = _run_command(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"])
    if lsof_result is None:
        raise RuntimeError("Neither `ss` nor `lsof` is available to inspect listening ports.")

    return sorted(
        {int(line.strip()) for line in lsof_result.stdout.splitlines() if line.strip().isdigit()}
    )


def listening_pids(port: int) -> list[int]:
    if os.name == "nt":
        return windows_listening_pids(port)
    return posix_listening_pids(port)


def kill_process_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        return

    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def kill_listener_pid(pid: int, force: bool = False) -> None:
    if os.name == "nt":
        result = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return

        output = " ".join(part.strip() for part in (result.stdout, result.stderr) if part).strip()
        lowered_output = output.lower()

        if "not found" in lowered_output or "non trovato" in lowered_output:
            return

        if "access is denied" in lowered_output or "accesso negato" in lowered_output:
            raise RuntimeError(
                f"Cannot kill PID {pid}: access denied. "
                "Run the command from an Administrator PowerShell, or stop the owning elevated/WSL/Docker process first."
            )

        raise RuntimeError(f"Failed to kill PID {pid}: {output or 'taskkill returned an error.'}")
        return

    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return


def kill_port_listeners(port: int) -> list[int]:
    pids = listening_pids(port)
    if not pids:
        print(f"No process is listening on port {port}.")
        return []

    print(f"Stopping listener(s) on port {port}: {pids}")
    for pid in pids:
        kill_listener_pid(pid)

    deadline = time.monotonic() + PORT_RELEASE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        remaining = listening_pids(port)
        if not remaining:
            return pids

        if os.name != "nt":
            for pid in remaining:
                kill_listener_pid(pid, force=True)

        time.sleep(0.2)

    remaining = listening_pids(port)
    if remaining:
        raise RuntimeError(f"Port {port} is still in use by PID(s): {remaining}")

    return pids


def ensure_port_is_available(port: int) -> None:
    if os.name != "nt" or not is_port_in_use(port):
        return

    pids = windows_listening_pids(port)
    if not pids:
        raise RuntimeError(
            f"Port {port} is already in use, but the owning PID could not be resolved."
        )

    print(f"Port {port} is already in use by PID(s) {pids}. Stopping stale process tree(s)...")
    for pid in pids:
        kill_process_tree(pid)

    deadline = time.monotonic() + PORT_RELEASE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if not is_port_in_use(port):
            return
        time.sleep(0.2)

    raise RuntimeError(
        f"Port {port} is still busy after cleanup. "
        "Run `taskkill /PID <pid> /T /F` or restart the terminal."
    )


def launch_uvicorn() -> subprocess.Popen[bytes]:
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        APP_PATH,
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--reload",
    ]
    kwargs: dict[str, object] = {}

    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    return subprocess.Popen(command, **kwargs)


def run_migrations() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")

    if result.returncode != 0:
        raise RuntimeError("Database migrations failed. See Alembic output above.")


def stop_uvicorn(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return

    kill_process_tree(process.pid)

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backend dev server helper.")
    parser.add_argument(
        "--kill-port",
        type=int,
        help="Kill only the process or processes currently listening on this port, then exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.kill_port is not None:
        try:
            kill_port_listeners(args.kill_port)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    try:
        ensure_port_is_available(PORT)
        run_migrations()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    process = launch_uvicorn()

    try:
        return process.wait()
    except KeyboardInterrupt:
        print("\nStopping backend dev server...")
        stop_uvicorn(process)
        return 130
    finally:
        stop_uvicorn(process)


if __name__ == "__main__":
    raise SystemExit(main())
