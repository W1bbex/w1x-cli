import threading

from w1x import models


def test_cancelled_inflight_preload_is_unloaded(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    loaded = set()

    def generate(**kwargs):
        if kwargs.get("keep_alive") == 0:
            loaded.discard(kwargs["model"])
        else:
            entered.set()
            assert release.wait(5)
            loaded.add(kwargs["model"])

    monkeypatch.setattr(models, "generate", generate)
    monkeypatch.setattr(models, "running_models", lambda: list(loaded))
    models.start_background_preload("offline:latest")
    task = models._CURRENT_PRELOAD
    try:
        assert entered.wait(5)
        models.reset_all_models()
    finally:
        release.set()
        task.thread.join(5)
    assert not task.thread.is_alive()
    assert loaded == set()


def test_replacement_model_survives_old_preload_completion(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    loaded = set()

    def generate(**kwargs):
        name = kwargs["model"]
        if kwargs.get("keep_alive") == 0:
            loaded.discard(name)
        else:
            if name == "old:latest":
                entered.set()
                assert release.wait(5)
            loaded.add(name)

    monkeypatch.setattr(models, "generate", generate)
    models.start_background_preload("old:latest")
    old = models._CURRENT_PRELOAD
    try:
        assert entered.wait(5)
        models.start_background_preload("new:latest")
        new = models._CURRENT_PRELOAD
        assert new.event.wait(5)
    finally:
        release.set()
        old.thread.join(5)
    assert loaded == {"new:latest"}


def test_same_model_replacement_is_not_unloaded(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def generate(**kwargs):
        calls.append(kwargs.get("keep_alive"))
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)

    monkeypatch.setattr(models, "generate", generate)
    monkeypatch.setattr(models, "running_models", lambda: [])
    models.start_background_preload("shared:latest")
    old = models._CURRENT_PRELOAD
    try:
        assert entered.wait(5)
        models.reset_all_models()
        models.start_background_preload("shared:latest")
        new = models._CURRENT_PRELOAD
    finally:
        release.set()
        old.thread.join(5)
    assert new.event.wait(5)
    assert len(calls) == 2
    assert 0 not in calls


def test_same_model_replacement_cancelled_before_start_unloads_inherited(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    loaded = set()

    def generate(**kwargs):
        name = kwargs["model"]
        if kwargs.get("keep_alive") == 0:
            loaded.discard(name)
        else:
            entered.set()
            assert release.wait(5)
            loaded.add(name)

    monkeypatch.setattr(models, "generate", generate)
    monkeypatch.setattr(models, "running_models", lambda: [])
    models.start_background_preload("shared:latest")
    old = models._CURRENT_PRELOAD
    try:
        assert entered.wait(5)
        models.reset_all_models()
        models.start_background_preload("shared:latest")
        new = models._CURRENT_PRELOAD
        # Cancel the replacement while `old` still holds the model lock.
        models.reset_all_models()
    finally:
        release.set()
        old.thread.join(5)
    assert new.event.wait(5)
    assert loaded == set()
