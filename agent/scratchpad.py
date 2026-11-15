"""外部草稿纸（scratchpad）：把「做了什么、结果如何」写成本地文件。

为什么不写进记事本：程序要读回记事本里的文字，必须截图再让 7B 认字
（10 秒以上一次，还可能认错），而且每次读写要切窗口，一个任务切十几次
就白白浪费半分钟，切换过程本身还可能点错。读写本地文件是毫秒级，
零切换、零识别错误。记事本适合给人看，草稿纸适合给程序看。

它解决两件事：
1. 上下文不够 —— 7B 只有 4096 token，全程历史塞不进去。每步只回填
   最近几条摘要，模型就不会「忘了自己已经打开过窗口」而重复点开始菜单。
2. 可复盘 —— 一个任务一个文件，出问题能顺着看它当时到底在想什么。
   每次 start() 新建文件，天然不会和上一个任务混在一起。
"""
import os
import time


class Scratchpad:
    MAX_FILES = 10  # 只留最近 10 个任务记录，多了自动清，避免堆积

    def __init__(self, root="data/scratch"):
        self.root = root
        self.path = None
        self.lines = []

    def start(self, task: str):
        """开始一个新任务：新建独立文件，绝不追加到上次的文件里。"""
        os.makedirs(self.root, exist_ok=True)
        base = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(self.root, base + ".md")
        i = 2
        while os.path.exists(path):   # 同一秒内连开两个任务也不能覆盖
            path = os.path.join(self.root, f"{base}-{i}.md")
            i += 1
        self.path = path
        self.lines = []
        self._write(f"# 任务：{task}\n\n开始时间：{time.strftime('%H:%M:%S')}\n\n")
        self._prune()

    def note(self, step: int, did: str, result: str = ""):
        """记一步：做了什么、结果如何。"""
        line = f"{step}. {did}" + (f" → {result}" if result else "") + "\n"
        self.lines.append(line)
        self._write(line)

    def recent(self, n: int = 6) -> str:
        """最近 n 步，回填给模型当「工作记忆」。"""
        return "".join(self.lines[-n:])

    def _write(self, text: str):
        if not self.path:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass  # 草稿纸写不进去不能影响主流程

    def _prune(self):
        """只保留最近 MAX_FILES 个任务记录。"""
        try:
            files = sorted(f for f in os.listdir(self.root) if f.endswith(".md"))
            for old in files[:-self.MAX_FILES]:
                os.remove(os.path.join(self.root, old))
        except Exception:
            pass
