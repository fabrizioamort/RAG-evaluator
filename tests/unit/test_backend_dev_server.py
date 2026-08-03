from __future__ import annotations

import sys
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


def load_dev_server_module():
    module_path = Path(__file__).resolve().parents[2] / "platform" / "backend" / "dev_server.py"
    spec = spec_from_file_location("backend_dev_server", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load module from {module_path}")

    backend_path = str(module_path.parent)
    path_was_present = backend_path in sys.path
    if not path_was_present:
        sys.path.insert(0, backend_path)

    try:
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if not path_was_present:
            sys.path.remove(backend_path)


class DevServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dev_server = load_dev_server_module()

    def test_posix_listening_pids_returns_unique_listener_pids_from_ss(self) -> None:
        mock_result = SimpleNamespace(
            returncode=0,
            stdout=(
                "State  Recv-Q Send-Q Local Address:Port Peer Address:PortProcess\n"
                'LISTEN 0      4096   127.0.0.1:8000      0.0.0.0:*    users:(("python",pid=4242,fd=3))\n'
                'LISTEN 0      4096   0.0.0.0:8000        0.0.0.0:*    users:(("python",pid=5151,fd=4))\n'
            ),
            stderr="",
        )

        with patch.object(self.dev_server.subprocess, "run", return_value=mock_result):
            pids = self.dev_server.posix_listening_pids(8000)

        self.assertEqual(pids, [4242, 5151])

    def test_windows_listening_pids_returns_unique_listener_pids(self) -> None:
        mock_result = SimpleNamespace(
            stdout=(
                "  TCP    127.0.0.1:8000      0.0.0.0:0      LISTENING       4242\n"
                "  TCP    127.0.0.1:8000      127.0.0.1:50123 ESTABLISHED   4242\n"
                "  TCP    127.0.0.1:8000      0.0.0.0:0      LISTENING       5151\n"
            )
        )

        with patch.object(self.dev_server.subprocess, "run", return_value=mock_result):
            pids = self.dev_server.windows_listening_pids(8000)

        self.assertEqual(pids, [4242, 5151])

    def test_ensure_port_is_available_kills_windows_listener_and_waits_for_release(self) -> None:
        with (
            patch.object(self.dev_server.os, "name", "nt"),
            patch.object(self.dev_server, "is_port_in_use", side_effect=[True, False]),
            patch.object(self.dev_server, "windows_listening_pids", return_value=[4242]),
            patch.object(self.dev_server, "kill_process_tree") as kill_process_tree,
            patch.object(self.dev_server.time, "sleep") as sleep,
        ):
            self.dev_server.ensure_port_is_available(8000)

        kill_process_tree.assert_called_once_with(4242)
        sleep.assert_not_called()

    def test_stop_uvicorn_terminates_running_process_tree(self) -> None:
        process = Mock()
        process.pid = 4242
        process.poll.return_value = None

        with patch.object(self.dev_server, "kill_process_tree") as kill_process_tree:
            self.dev_server.stop_uvicorn(process)

        kill_process_tree.assert_called_once_with(4242)
        process.wait.assert_called_once_with(timeout=5)

    def test_main_kill_port_mode_cleans_port_and_exits_without_launching_server(self) -> None:
        with (
            patch.object(
                self.dev_server, "kill_port_listeners", return_value=[4242]
            ) as kill_port_listeners,
            patch.object(self.dev_server, "launch_uvicorn") as launch_uvicorn,
        ):
            exit_code = self.dev_server.main(["--kill-port", "8000"])

        self.assertEqual(exit_code, 0)
        kill_port_listeners.assert_called_once_with(8000)
        launch_uvicorn.assert_not_called()

    def test_main_runs_migrations_before_launching_server(self) -> None:
        process = Mock()
        process.wait.return_value = 0
        process.poll.return_value = 0

        with (
            patch.object(self.dev_server, "ensure_port_is_available") as ensure_port,
            patch.object(self.dev_server, "run_migrations") as run_migrations,
            patch.object(self.dev_server, "launch_uvicorn", return_value=process) as launch_uvicorn,
        ):
            exit_code = self.dev_server.main([])

        self.assertEqual(exit_code, 0)
        ensure_port.assert_called_once_with(8000)
        run_migrations.assert_called_once_with()
        launch_uvicorn.assert_called_once_with()

    def test_main_exits_when_migrations_fail(self) -> None:
        with (
            patch.object(self.dev_server, "ensure_port_is_available") as ensure_port,
            patch.object(
                self.dev_server, "run_migrations", side_effect=RuntimeError("migration failed")
            ) as run_migrations,
            patch.object(self.dev_server, "launch_uvicorn") as launch_uvicorn,
        ):
            exit_code = self.dev_server.main([])

        self.assertEqual(exit_code, 1)
        ensure_port.assert_called_once_with(8000)
        run_migrations.assert_called_once_with()
        launch_uvicorn.assert_not_called()

    def test_kill_listener_pid_reports_windows_access_denied(self) -> None:
        mock_result = SimpleNamespace(returncode=1, stdout="", stderr="ERROR: Access is denied")

        with (
            patch.object(self.dev_server.os, "name", "nt"),
            patch.object(self.dev_server.subprocess, "run", return_value=mock_result),
        ):
            with self.assertRaisesRegex(RuntimeError, "Administrator PowerShell"):
                self.dev_server.kill_listener_pid(4242)


if __name__ == "__main__":
    unittest.main()
