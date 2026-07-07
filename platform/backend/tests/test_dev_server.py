"""Tests for the development server helper."""

from types import SimpleNamespace

import dev_server


def test_ensure_development_paths_creates_sqlite_parent(tmp_path, monkeypatch):
    storage_path = tmp_path / "storage"
    database_path = tmp_path / "nested" / "dev.db"

    monkeypatch.setattr(
        dev_server,
        "settings",
        SimpleNamespace(
            STORAGE_PATH=str(storage_path),
            DATABASE_URL=f"sqlite+aiosqlite:///{database_path.as_posix()}",
            is_sqlite=True,
        ),
    )

    dev_server.ensure_development_paths()

    assert storage_path.is_dir()
    assert database_path.parent.is_dir()
    assert not database_path.exists()
