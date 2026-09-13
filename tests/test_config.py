from support_agent import config


def test_paths_are_under_root():
    assert config.DATA_DIR == config.ROOT / "data"
    assert config.brand_dir() == config.PROCESSED_DIR / config.BRAND


def test_offline_flag(monkeypatch):
    monkeypatch.setenv("LLM_OFFLINE", "1")
    assert config.offline() is True
    monkeypatch.setenv("LLM_OFFLINE", "0")
    assert config.offline() is False
