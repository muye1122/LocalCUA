#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""配置加载：读取项目根目录的 config.yaml（不存在则回退空配置）。"""

import os
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def project_root() -> Path:
    """agent/ 的上级目录即项目根。"""
    return Path(__file__).resolve().parent.parent


def load_config(path=None) -> dict:
    """按以下顺序查找配置，第一个命中即用：
    1) 显式 path 参数
    2) 环境变量 LOCALCUA_CONFIG 指向的文件
    3) 项目根目录 config.yaml（源码 / 开发模式，可被用户编辑）
    4) 打包进 wheel 的 agent/config.yaml（pip 安装后）
    """
    candidates = []
    if path:
        candidates.append(Path(path))
    env = os.environ.get("LOCALCUA_CONFIG")
    if env:
        candidates.append(Path(env))
    candidates.append(project_root() / "config.yaml")
    try:
        # 安装模式下，配置随包一起发布，用 importlib.resources 读取
        import importlib.resources as ir
        candidates.append(ir.files("agent") / "config.yaml")
    except Exception:
        pass

    for cand in candidates:
        try:
            if cand and Path(cand).exists():
                if yaml is None:  # pragma: no cover
                    raise RuntimeError("未安装 pyyaml，无法读取 config.yaml，请先 pip install pyyaml")
                with open(cand, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
        except RuntimeError:
            raise
        except Exception:
            continue
    raise FileNotFoundError(
        "未找到 config.yaml。请将项目根目录的 config.yaml 复制过来，"
        "或用环境变量 LOCALCUA_CONFIG 指定路径。"
    )
