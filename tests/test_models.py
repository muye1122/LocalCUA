"""模型客户端测试：data URL 编码、配置解析、环境变量覆盖（不触发网络）。"""

import os

from PIL import Image

from agent.models import ModelClient, image_to_data_url
from agent.config import load_config


def test_image_to_data_url():
    img = Image.new("RGB", (4, 4), (255, 0, 0))
    u = image_to_data_url(img)
    assert u.startswith("data:image/png;base64,")
    assert len(u) > 50


def test_modelclient_ollama_default(monkeypatch):
    monkeypatch.setenv("CU_MODEL", "")
    monkeypatch.setenv("CU_PROVIDER", "")
    cfg = {"provider": {"type": "ollama", "ollama": {"model": "qwen2.5vl:7b"}}}
    mc = ModelClient(cfg)
    assert mc.type == "ollama"
    assert mc.model == "qwen2.5vl:7b"


def test_modelclient_env_override(monkeypatch):
    monkeypatch.setenv("CU_MODEL", "llava:latest")
    cfg = {"provider": {"type": "ollama", "ollama": {"model": "qwen2.5vl:7b"}}}
    mc = ModelClient(cfg)
    assert mc.model == "llava:latest"


def test_modelclient_openai(monkeypatch):
    monkeypatch.setenv("CU_OPENAI_KEY", "")
    cfg = {
        "provider": {
            "type": "openai",
            "openai": {"model": "gpt-4o", "api_key": "sk-test"},
        }
    }
    mc = ModelClient(cfg)
    assert mc.type == "openai"
    assert mc.api_key == "sk-test"


def test_load_config_returns_dict():
    # 项目根目录存在 config.yaml，应能解析为 dict
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert "provider" in cfg
