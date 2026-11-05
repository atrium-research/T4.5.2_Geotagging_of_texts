import os
import time

from app import main


def _make_runs(root):
    old, new = root / "aaaaaaaaaaaa", root / "bbbbbbbbbbbb"
    old.mkdir(); new.mkdir()
    long_ago = time.time() - main.UNSAVED_RUN_TTL_SECONDS - 60
    os.utime(old, (long_ago, long_ago))
    return old, new


def test_unsaved_runs_older_than_an_hour_are_deleted(tmp_path, monkeypatch):
    old, new = _make_runs(tmp_path)
    monkeypatch.setattr(main, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(main, "SAVE_RUNS", False)
    main._purge_unsaved_runs()
    assert not old.exists() and new.exists()


def test_saved_runs_are_never_deleted(tmp_path, monkeypatch):
    old, new = _make_runs(tmp_path)
    monkeypatch.setattr(main, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(main, "SAVE_RUNS", True)
    main._purge_unsaved_runs()
    assert old.exists() and new.exists()
