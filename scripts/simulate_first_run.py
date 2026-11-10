"""Run the CLI with temporary settings and an initially empty model list."""
import sys
import tempfile
from pathlib import Path


def main():
    app_dir = Path(__file__).resolve().parents[1]
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))

    import w1x.config as config
    import w1x.models as models

    path_names = ("DATA_DIR", "CHAT_DIR", "CURRENT_CHAT_PATH", "FACTS_PATH", "CONFIG_FILE_PATH")
    original_paths = {name: getattr(config, name) for name in path_names}
    installed_models = models.installed_models

    with tempfile.TemporaryDirectory(prefix="w1x-firstrun-") as directory:
        sandbox = Path(directory)
        config.DATA_DIR = sandbox
        config.CHAT_DIR = sandbox / "chats"
        config.CURRENT_CHAT_PATH = sandbox / "current-chat.json"
        config.FACTS_PATH = sandbox / "memory-facts.json"
        config.CONFIG_FILE_PATH = sandbox / "config.json"

        def initially_empty_models():
            return installed_models() if config.CONFIG_FILE_PATH.exists() else []

        models.installed_models = initially_empty_models
        try:
            from w1x.app import main as run_app
            run_app()
        finally:
            for name, path in original_paths.items():
                setattr(config, name, path)
            models.installed_models = installed_models


if __name__ == "__main__":
    main()
