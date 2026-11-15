# tools/legacy — 开发期的一次性真机实验脚本

这里是 2026-09~11 开发过程中留下的 24 个探针 / 对照实验脚本。
**主程序不依赖本目录下任何文件**；`tools/` 主目录才放别人会用到的实用工具
（跑任务、自检、下载权重与模型、沙箱、轨迹分析等）。

## 为什么留着而不是删掉

README 的「技术复盘：真机踩坑链」里那些数字，都是这些脚本跑出来的：

| 脚本 | 对应的结论 |
|---|---|
| `try_locate.py` / `two_step.py` / `nine_grid_test.py` | 定位 16.2s/次 → 0.0s/次、9 秒花在逐个起名、小模型估坐标误差 ±30~90px |
| `caption_test.py` / `select_test.py` / `text_select_test.py` | 「让模型给元素起名可靠、让它选编号不可靠」 |
| `code_lab.py` / `try_devcpp.py` / `try_code_task.py` | C 代码「写 → 编译 → 读报错 → 自己改」闭环 |
| `try_dblclick.py` / `try_desktop.py` | 单击改双击、桌面图标优先（读 SysListView32） |
| `smoke_grounding.py` / `icon_detect.py` / `grid_probe.py` | YOLO 检测 + 8B 起名方案的原型验证 |
| `ab_test.py` / `try_batch.py` / `try_coder.py` | 提示词写法、动作序列、coder 模型的对照实验 |

它们也是「这个项目是自己一步步真机试出来的」最直接的证据。

## 运行注意

- 从**项目根目录**运行，例如 `python tools/legacy/try_locate.py`；
  脚本里的 `sys.path` 已按多出的一层调整过。
- 这些脚本写于早期 7B 模型时期，**文中的实验结论针对当时的模型**；
  脚本里的模型名已统一为当前的 `qwen3vl:8b` 以便能跑，
  但结论数字仍是当时实测，别拿它们当当前版本的性能数据。
- 部分脚本默认只读不点鼠标，带 `--click` 之类参数才会真的动键鼠，跑前先看用法。
- 有些依赖 `data/` 下的历史截图 / 轨迹，公开仓库不含这些数据，
  所以那些脚本只能看代码、不能直接复现数值。
