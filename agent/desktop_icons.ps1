# 读取桌面图标（名字 + 屏幕坐标），零模型调用、零猜测。
#
# 为什么不用视觉：桌面有十几个图标，视觉定位要逐个给元素起名（每个都是一次
# 模型调用，真机实测这一步就烧掉 30 秒以上），而桌面本身就是一个标准的
# SysListView32 控件 —— 直接问 Windows 要"第几个图标叫什么、在哪"就是
# 毫秒级且 ±0px 的答案。
#
# 输出：UTF-8 JSON 单行
#   {"ok":true,"screen_w":2560,"items":[{"name":"Red Panda Dev-C++","x":120,"y":80,"w":96,"h":96}]}
#   x/y 是图标格子中心的【物理像素屏幕坐标】；screen_w 是物理屏宽，调用方据此换算。

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false

Add-Type -TypeDefinition @'
using System;
using System.Text;
using System.Collections.Generic;
using System.Runtime.InteropServices;

public class DeskIconReader
{
    const uint LVM_FIRST        = 0x1000;
    const uint LVM_GETITEMCOUNT = LVM_FIRST + 4;
    const uint LVM_GETITEMRECT  = LVM_FIRST + 14;
    const uint LVM_GETITEMTEXTW = LVM_FIRST + 115;

    [StructLayout(LayoutKind.Sequential)]
    public struct LVITEMW
    {
        public uint mask;
        public int iItem;
        public int iSubItem;
        public uint state;
        public uint stateMask;
        public IntPtr pszText;
        public int cchTextMax;
        public int iImage;
        public IntPtr lParam;
        public int iIndent;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct RECT { public int left, top, right, bottom; }

    [StructLayout(LayoutKind.Sequential)]
    public struct POINT { public int x, y; }

    public delegate bool EnumProc(IntPtr hWnd, IntPtr lParam);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    public static extern IntPtr FindWindow(string cls, string win);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    public static extern IntPtr FindWindowEx(IntPtr parent, IntPtr child, string cls, string win);
    [DllImport("user32.dll")]
    public static extern bool EnumWindows(EnumProc cb, IntPtr lParam);
    [DllImport("user32.dll", EntryPoint = "SendMessageW")]
    public static extern IntPtr SendMessage(IntPtr hWnd, uint msg, IntPtr wParam, IntPtr lParam);
    [DllImport("user32.dll")]
    public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint pid);
    [DllImport("user32.dll")]
    public static extern bool ClientToScreen(IntPtr hWnd, ref POINT p);
    [DllImport("user32.dll")]
    public static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")]
    public static extern int GetSystemMetrics(int idx);
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern IntPtr OpenProcess(uint access, bool inherit, uint pid);
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool CloseHandle(IntPtr h);
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern IntPtr VirtualAllocEx(IntPtr hp, IntPtr addr, IntPtr size, uint type, uint protect);
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool VirtualFreeEx(IntPtr hp, IntPtr addr, IntPtr size, uint type);
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool WriteProcessMemory(IntPtr hp, IntPtr addr, byte[] buf, IntPtr size, out IntPtr written);
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool ReadProcessMemory(IntPtr hp, IntPtr addr, byte[] buf, IntPtr size, out IntPtr read);

    static IntPtr FindListView()
    {
        IntPtr progman = FindWindow("Progman", null);
        IntPtr defview = IntPtr.Zero;
        if (progman != IntPtr.Zero)
            defview = FindWindowEx(progman, IntPtr.Zero, "SHELLDLL_DefView", null);
        if (defview == IntPtr.Zero)
        {
            // 换了壁纸/多显示器时，SHELLDLL_DefView 会挪到某个 WorkerW 下面
            IntPtr found = IntPtr.Zero;
            EnumWindows(delegate(IntPtr h, IntPtr l)
            {
                IntPtr v = FindWindowEx(h, IntPtr.Zero, "SHELLDLL_DefView", null);
                if (v != IntPtr.Zero) { found = v; return false; }
                return true;
            }, IntPtr.Zero);
            defview = found;
        }
        if (defview == IntPtr.Zero) return IntPtr.Zero;
        IntPtr lv = FindWindowEx(defview, IntPtr.Zero, "SysListView32", "FolderView");
        if (lv == IntPtr.Zero) lv = FindWindowEx(defview, IntPtr.Zero, "SysListView32", null);
        return lv;
    }

    static string Esc(string s)
    {
        if (s == null) return "";
        StringBuilder sb = new StringBuilder();
        foreach (char c in s)
        {
            if (c == '"' || c == '\\') sb.Append('\\').Append(c);
            else if (c == '\n') sb.Append("\\n");
            else if (c == '\r') sb.Append("\\r");
            else if (c == '\t') sb.Append("\\t");
            else if (c < 0x20) sb.Append(' ');
            else sb.Append(c);
        }
        return sb.ToString();
    }

    public static string Run()
    {
        SetProcessDPIAware();
        int sw = GetSystemMetrics(0);   // SM_CXSCREEN，DPI aware 下就是物理像素
        int sh = GetSystemMetrics(1);
        IntPtr lv = FindListView();
        if (lv == IntPtr.Zero)
            return "{\"ok\":false,\"error\":\"no desktop listview\",\"screen_w\":" + sw + ",\"screen_h\":" + sh + ",\"items\":[]}";

        uint pid;
        GetWindowThreadProcessId(lv, out pid);
        IntPtr hp = OpenProcess(0x1F0FFF, false, pid);
        if (hp == IntPtr.Zero)
            return "{\"ok\":false,\"error\":\"openprocess failed\",\"screen_w\":" + sw + ",\"screen_h\":" + sh + ",\"items\":[]}";

        IntPtr remote = VirtualAllocEx(hp, IntPtr.Zero, (IntPtr)8192, 0x3000, 0x04);
        if (remote == IntPtr.Zero) { CloseHandle(hp); return "{\"ok\":false,\"error\":\"alloc failed\",\"items\":[]}"; }

        int count = (int)SendMessage(lv, LVM_GETITEMCOUNT, IntPtr.Zero, IntPtr.Zero);
        long baseAddr = remote.ToInt64();
        long nameAddr = baseAddr + 1024;
        long rectAddr = baseAddr + 2048;
        StringBuilder json = new StringBuilder();
        json.Append("{\"ok\":true,\"screen_w\":").Append(sw).Append(",\"screen_h\":").Append(sh).Append(",\"items\":[");
        int emitted = 0;
        IntPtr w = IntPtr.Zero;

        for (int i = 0; i < count && i < 500; i++)
        {
            LVITEMW it = new LVITEMW();
            it.mask = 0x0001;              // LVIF_TEXT
            it.iItem = i;
            it.iSubItem = 0;
            it.pszText = (IntPtr)nameAddr;
            it.cchTextMax = 260;

            int sz = Marshal.SizeOf(typeof(LVITEMW));
            byte[] ib = new byte[sz];
            IntPtr pIt = Marshal.AllocHGlobal(sz);
            Marshal.StructureToPtr(it, pIt, false);
            Marshal.Copy(pIt, ib, 0, sz);
            Marshal.FreeHGlobal(pIt);

            if (!WriteProcessMemory(hp, remote, ib, (IntPtr)sz, out w)) continue;
            SendMessage(lv, LVM_GETITEMTEXTW, (IntPtr)i, remote);

            byte[] tb = new byte[520];
            if (!ReadProcessMemory(hp, (IntPtr)nameAddr, tb, (IntPtr)tb.Length, out w)) continue;
            string name = Encoding.Unicode.GetString(tb);
            int z = name.IndexOf('\0');
            if (z >= 0) name = name.Substring(0, z);
            if (name.Length == 0) continue;

            byte[] zero = new byte[16];
            WriteProcessMemory(hp, (IntPtr)rectAddr, zero, (IntPtr)16, out w);
            SendMessage(lv, LVM_GETITEMRECT, (IntPtr)i, (IntPtr)rectAddr);
            byte[] rb = new byte[16];
            if (!ReadProcessMemory(hp, (IntPtr)rectAddr, rb, (IntPtr)16, out w)) continue;
            RECT rc = new RECT();
            rc.left = BitConverter.ToInt32(rb, 0);
            rc.top = BitConverter.ToInt32(rb, 4);
            rc.right = BitConverter.ToInt32(rb, 8);
            rc.bottom = BitConverter.ToInt32(rb, 12);
            if (rc.right <= rc.left || rc.bottom <= rc.top) continue;

            // 图标的"图标本体"在上半部分，文字标签在下半部分。
            // 双击要落在图标上（点文字一般也能选中，但点图标最稳）。
            int iconH = (rc.bottom - rc.top) * 5 / 9;
            POINT p = new POINT();
            p.x = (rc.left + rc.right) / 2;
            p.y = rc.top + iconH / 2;
            ClientToScreen(lv, ref p);

            if (emitted > 0) json.Append(',');
            json.Append("{\"name\":\"").Append(Esc(name)).Append("\",\"x\":").Append(p.x)
                .Append(",\"y\":").Append(p.y)
                .Append(",\"w\":").Append(rc.right - rc.left)
                .Append(",\"h\":").Append(rc.bottom - rc.top).Append('}');
            emitted++;
        }

        VirtualFreeEx(hp, remote, (IntPtr)0, 0x8000);
        CloseHandle(hp);
        json.Append("]}");
        return json.ToString();
    }
}
'@

# 先确保桌面可见（Minimized 状态下 ListView 也存在，坐标同样是屏幕坐标，
# 但被遮挡时双击会点到别的窗口上，所以调用方应先 Win+D）
Write-Output ([DeskIconReader]::Run())
