import runpy
from pathlib import Path

import pytest

from w1x import app, config, models


@pytest.mark.parametrize("app_fails", [False, True])
def test_first_run_simulator_is_import_safe_and_restores_state(monkeypatch, app_fails):
    names = ("DATA_DIR", "CHAT_DIR", "CURRENT_CHAT_PATH", "FACTS_PATH", "CONFIG_FILE_PATH")
    previous_paths = {name: getattr(config, name) for name in names}
    installed = lambda: ["existing-model"]
    monkeypatch.setattr(models, "installed_models", installed)
    script = Path(__file__).resolve().parents[1] / "scripts" / "simulate_first_run.py"

    namespace = runpy.run_path(str(script), run_name="first_run_simulator")

    assert {name: getattr(config, name) for name in names} == previous_paths
    assert models.installed_models is installed
    sandboxes = []

    def run_app():
        sandboxes.append(config.DATA_DIR)
        assert config.DATA_DIR.is_dir()
        assert config.is_first_run()
        assert models.installed_models() == []
        config.save_user_config({"configured": True})
        assert models.installed_models() == ["existing-model"]
        if app_fails:
            raise RuntimeError("simulated application failure")

    monkeypatch.setattr(app, "main", run_app)
    if app_fails:
        with pytest.raises(RuntimeError, match="simulated application failure"):
            namespace["main"]()
    else:
        namespace["main"]()

    assert len(sandboxes) == 1
    assert not sandboxes[0].exists()
    assert {name: getattr(config, name) for name in names} == previous_paths
    assert models.installed_models is installed
