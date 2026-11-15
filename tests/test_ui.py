"""Тесты для компонентов пользовательского интерфейса, пагинации и разрешения CLI-команд."""
from unittest.mock import MagicMock

from w1x import app as w1x_app
from w1x import catalog_ui, cli, commands, models_ui


def test_render_models_panel_pagination_threshold():
    """Проверка переключения между монолитным списком и пагинацией."""
    few_models = [f"model_{i}:latest" for i in range(5)]
    panel_few = models_ui.render_models_panel(few_models, 0, set())
    text_few = str(panel_few.renderable)
    assert "▲" not in text_few
    assert "▼" not in text_few
    assert "model_0" in text_few
    assert "model_4" in text_few

    many_models = [f"model_{i:02d}:latest" for i in range(25)]
    panel_many_top = models_ui.render_models_panel(many_models, 0, set())
    text_top = str(panel_many_top.renderable)
    assert "▲" not in text_top
    assert "▼" in text_top
    assert "model_00" in text_top

    panel_many_mid = models_ui.render_models_panel(many_models, 15, set())
    text_mid = str(panel_many_mid.renderable)
    assert "▲" in text_mid
    assert "▼" in text_mid


def test_interactive_config_context_tokens_invalid_warning(monkeypatch):
    """Проверка валидации и уведомления при вводе некорректного размера контекста."""
    prompts = iter(["3", "256", "0"])
    monkeypatch.setattr(models_ui.secondary_session, "prompt", lambda *a, **kw: next(prompts))
    monkeypatch.setattr(models_ui, "load_user_config", lambda: {"context_tokens": 8192})

    saved_cfgs = []
    monkeypatch.setattr(models_ui, "save_user_config", lambda cfg: saved_cfgs.append(cfg) or True)

    captured = []
    fake_console = MagicMock()
    fake_console.print = lambda *a, **kw: captured.append(" ".join(str(x) for x in a))
    monkeypatch.setattr(models_ui, "console", fake_console)

    models_ui.interactive_config_menu()
    assert not any("context_tokens" in cfg and cfg["context_tokens"] == 256 for cfg in saved_cfgs)
    assert any("не менее 512" in msg for msg in captured)


def test_handle_pull_cmd_variant_a_when_models_exist(monkeypatch):
    """Команда /pull при наличии моделей открывает каталог без перезаписи default_model."""
    catalog_called = False
    wait_called = False
    wizard_called = False

    def fake_catalog():
        nonlocal catalog_called
        catalog_called = True
        return "downloaded:7b"

    def fake_wait():
        nonlocal wait_called
        wait_called = True

    def fake_wizard():
        nonlocal wizard_called
        wizard_called = True
        return "wizard:4b"

    monkeypatch.setattr(commands.models_mod, "installed_models", lambda: ["existing:4b"])
    monkeypatch.setattr(catalog_ui, "interactive_model_catalog", fake_catalog)
    monkeypatch.setattr(catalog_ui, "wait_for_download_queue", fake_wait)
    monkeypatch.setattr(commands, "first_time_setup_wizard", fake_wizard)

    res_model = commands.handle_pull_cmd("", "existing:4b")
    assert catalog_called is True
    assert wait_called is True
    assert wizard_called is False
    assert res_model == "existing:4b"


def test_handle_pull_cmd_first_time_when_no_models(monkeypatch):
    """Команда /pull при 0 моделей запускает мастер первого запуска."""
    wizard_called = False

    def fake_wizard():
        nonlocal wizard_called
        wizard_called = True
        return "fresh:4b"

    monkeypatch.setattr(w1x_app, "installed_models", lambda: [])
    monkeypatch.setattr(commands.models_mod, "installed_models", lambda: [])
    monkeypatch.setattr(w1x_app, "first_time_setup_wizard", fake_wizard)
    monkeypatch.setattr(commands, "first_time_setup_wizard", fake_wizard)
    monkeypatch.setattr(commands, "start_background_preload", lambda _: None)
    monkeypatch.setattr(commands, "print_header", lambda _: None)

    res_model = commands.handle_pull_cmd("", "old:4b")
    assert wizard_called is True
    assert res_model == "fresh:4b"


def test_run_cli_oneshot_pull_routes_to_catalog_when_models_exist(monkeypatch):
    """w1x pull без аргументов при установленных моделях запускает каталог."""
    catalog_called = []
    wait_called = []

    def fake_catalog():
        catalog_called.append(True)
        return "m:4b"

    def fake_wait():
        wait_called.append(True)

    monkeypatch.setattr(catalog_ui, "interactive_model_catalog", fake_catalog)
    monkeypatch.setattr(catalog_ui, "wait_for_download_queue", fake_wait)

    code = cli.run_cli_oneshot(
        ["pull"],
        installed_models_fn=lambda: ["installed:4b"],
    )
    assert code == 0
    assert len(catalog_called) == 1
    assert len(wait_called) == 1


def test_run_cli_oneshot_multiword_commands_treated_as_llm_queries(monkeypatch):
    """Команды 'config ...' и 'unload ...' из нескольких слов перенаправляются в модель."""
    generated_prompts = []

    def fake_generate(model, messages, think_mode):
        generated_prompts.append(messages[-1]["content"])
        return "answer", None, False

    code = cli.run_cli_oneshot(
        ["config", "file", "location"],
        installed_models_fn=lambda: ["qwen:4b"],
        wait_for_model_ready_fn=lambda _: None,
        generate_answer_fn=fake_generate,
    )
    assert code == 0
    assert "config file location" in generated_prompts

    code2 = cli.run_cli_oneshot(
        ["unload", "memory", "query"],
        installed_models_fn=lambda: ["qwen:4b"],
        wait_for_model_ready_fn=lambda _: None,
        generate_answer_fn=fake_generate,
    )
    assert code2 == 0
    assert "unload memory query" in generated_prompts


def test_run_cli_oneshot_negative_number_expressions(monkeypatch):
    """Отрицательные числа не вызывают ошибку неизвестного флага CLI."""
    generated_prompts = []

    def fake_generate(model, messages, think_mode):
        generated_prompts.append(messages[-1]["content"])
        return "5", None, False

    code = cli.run_cli_oneshot(
        ["-5", "+", "10"],
        installed_models_fn=lambda: ["qwen:4b"],
        wait_for_model_ready_fn=lambda _: None,
        generate_answer_fn=fake_generate,
    )
    assert code == 0
    assert "-5 + 10" in generated_prompts
