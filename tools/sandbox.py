#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Windows Sandbox 助手：生成一个 .wsb 配置文件，把本 agent 目录只读映射进沙箱，
并在沙箱登录时自动启动 agent。这样 Agent 碰不到你真实系统的数据。

前置条件：
  - Windows 10/11 专业版或企业版
  - 已开启「Windows 沙箱」功能（控制面板 -> 程序 -> 启用或关闭 Windows 功能 -> 勾选 Windows Sandbox）
  - 沙箱内也要装好 Python 与依赖、Ollama 及模型（沙箱是临时环境，重启即清空）

用法：
  python tools/sandbox.py        # 在当前目录生成 agent_sandbox.wsb
  双击生成的 .wsb 即可启动隔离桌面，agent 会自动运行
"""

from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent


def write_wsb(path=None):
    path = Path(path) if path else (PROJECT / "agent_sandbox.wsb")
    agent_dir = PROJECT
    content = f"""<Configuration>
  <MappedFolders>
    <MappedFolder>
      <HostFolder>{agent_dir}</HostFolder>
      <ReadOnly>true</ReadOnly>
    </MappedFolder>
  </MappedFolders>
  <LogonCommand>
    <Command>powershell -NoExit -Command "Set-Location '{agent_dir}'; python run.py"</Command>
  </LogonCommand>
</Configuration>
"""
    path.write_text(content, encoding="utf-8")
    return path


if __name__ == "__main__":
    print("已生成 Windows Sandbox 配置：", write_wsb())
