# UIA element enumerator (ASCII only - PowerShell parses this file as GBK on zh-CN)
# Usage: powershell -NoProfile -File uia_enum.ps1 [maxElements] [hwndOverride]
# Output: single JSON array on stdout, UTF-8: [{name,type,l,t,r,b},...]
param(
    [int]$Max = 200,
    [long]$Hwnd = 0,
    [switch]$List
)
$ErrorActionPreference = 'SilentlyContinue'
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Add-Type @"
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
public static class Native {
    public delegate bool EnumCb(IntPtr h, IntPtr l);
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowTextW(IntPtr h, StringBuilder s, int n);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetClassNameW(IntPtr h, StringBuilder s, int n);
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumCb cb, IntPtr l);
    public static List<long> VisibleWindows() {
        var list = new List<long>();
        EnumWindows((h, l) => {
            if (IsWindowVisible(h)) {
                var t = new StringBuilder(256); GetWindowTextW(h, t, 256);
                if (t.Length > 0) list.Add(h.ToInt64());
            }
            return true;
        }, IntPtr.Zero);
        return list;
    }
}
"@
[void][Native]::SetProcessDPIAware()

if ($List) {
    $wins = @()
    foreach ($h64 in [Native]::VisibleWindows()) {
        $h = [IntPtr]$h64
        $t = New-Object System.Text.StringBuilder 256; [void][Native]::GetWindowTextW($h, $t, 256)
        $c = New-Object System.Text.StringBuilder 256; [void][Native]::GetClassNameW($h, $c, 256)
        $wins += @{ hwnd = $h64; cls = $c.ToString(); title = $t.ToString() }
    }
    ConvertTo-Json -InputObject $wins -Compress -Depth 2
    exit
}

$h = $Hwnd
if ($h -eq 0) { $h = [Native]::GetForegroundWindow() }
if ($h -eq [IntPtr]::Zero -or -not [Native]::IsWindowVisible($h)) { echo '[]'; exit }

$sb = New-Object System.Text.StringBuilder 512
[void][Native]::GetWindowTextW($h, $sb, 512)
$winTitle = $sb.ToString()

try { $win = [System.Windows.Automation.AutomationElement]::FromHandle($h) } catch { echo '[]'; exit }
if ($null -eq $win) { echo '[]'; exit }

$all = $win.FindAll([System.Windows.Automation.TreeScope]::Descendants, [System.Windows.Automation.Condition]::TrueCondition)
$out = New-Object System.Collections.Generic.List[object]
$n = $all.Count
$cap = [Math]::Min($Max, $n)
for ($i = 0; $i -lt $cap; $i++) {
    $e = $all[$i]
    $cur = $e.Current
    if ($cur.IsOffscreen) { continue }
    $name = $cur.Name
    $type = $cur.LocalizedControlType
    $aid = $cur.AutomationId
    $r = $cur.BoundingRectangle
    if ($r.Width -le 1 -or $r.Height -le 1) { continue }
    if ([string]::IsNullOrWhiteSpace($name) -and [string]::IsNullOrWhiteSpace($aid)) { continue }
    $out.Add(@{
        name = $name
        type = $type
        id   = $aid
        l = [int][Math]::Round($r.X);  t = [int][Math]::Round($r.Y)
        r = [int][Math]::Round($r.Right); b = [int][Math]::Round($r.Bottom)
    })
}
$meta = @{ window = $winTitle; count = $n }
$payload = @{ meta = $meta; elements = $out }
ConvertTo-Json -InputObject $payload -Compress -Depth 3
