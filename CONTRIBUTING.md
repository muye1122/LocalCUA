# 贡献指南

感谢对 LocalCUA 感兴趣！下面是本地开发与提交流程。

## 开发环境

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## 运行测试

```bash
python -m pytest tests -q -p no:cacheprovider
```

测试不依赖真实显示器：`conftest.py` 已把键鼠、截图、输入法全部隔离为假对象。
CI 在 Ubuntu 上只做 `compileall` 语法检查（全量 GUI 测试需 Windows 桌面与本机 Ollama，
Linux runner 无法如实复现）；本地用 `python -m pytest -q` 跑完整的 281 项。

## 代码约定

- 模块保持**可导入、可测试**：涉及 `pyautogui` 的导入请懒加载（参考 `agent/safety.py`、`agent/core.py`）。
- 新增能力尽量用 `config.yaml` 开关暴露，默认保持开箱即用。
- 纯逻辑改动请补/改 `tests/`，并保证 `python -m pytest tests` 全绿。
- 提交信息用中文或英文均可，说明「为什么」比「改了什么」更重要。

## 提交 PR

1. Fork 并新建分支（`feat/xxx`、`fix/xxx`）。
2. 确保本地测试通过。
3. 在 PR 描述里说明改动动机与验证方式。

## 方向建议

- 新的 grounding 后端（无障碍树、更强检测器）
- 校验/规划提示词的调优
- 轨迹可视化与基准评估
- 多语言界面
