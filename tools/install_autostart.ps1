# 分析系统开机自启配置：登录触发 + 崩溃 1 分钟自动拉起。
# 用法（双击 install_autostart.bat 或）：
#   powershell -NoProfile -ExecutionPolicy Bypass -File install_autostart.ps1          # 安装
#   powershell -NoProfile -ExecutionPolicy Bypass -File install_autostart.ps1 -Remove  # 卸载
# 需要：与脚本同目录存在 分析系统.exe；管理员权限（脚本自动请求 UAC）。
# 编码注意：本文件必须保存为带 BOM 的 UTF-8（Windows PowerShell 5.1 依赖 BOM 识别）。

param(
    [switch]$Remove,
    [string]$LogFile = ""
)

$ErrorActionPreference = "Stop"
$taskName = "分析系统开机自启"
$output = New-Object System.Collections.Generic.List[string]

function Say($msg) {
    $output.Add($msg)
    Write-Host $msg
}

# ---- 管理员自提权（结果经日志文件回传给原进程）----
$identity = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $identity.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    $log = Join-Path $PSScriptRoot "autostart_result.txt"
    Remove-Item $log -ErrorAction SilentlyContinue
    $argline = if ($Remove) { "-File `"$PSCommandPath`" -Remove -LogFile `"$log`"" }
               else { "-File `"$PSCommandPath`" -LogFile `"$log`"" }
    Start-Process powershell -Verb RunAs -Wait -ArgumentList $argline
    if (Test-Path $log) { Get-Content $log }
    else { Write-Host "[ERROR] 提权未完成（UAC 被取消？）" -ForegroundColor Red }
    exit
}

$exe = Join-Path $PSScriptRoot "分析系统.exe"
if (-not (Test-Path $exe)) {
    Say "[ERROR] 未找到 $exe"
}
elseif ($Remove) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    Say "[OK] 已卸载计划任务：$taskName"
}
else {
    # ---- 注册计划任务 ----
    # GUI 程序必须"用户登录时"运行（InteractiveToken），不能 SYSTEM（会话不可见）；
    # RestartCount：进程崩溃后 1 分钟自动拉起；ExecutionTimeLimit 0 = 不限运行时长
    # （默认 72h 会被强杀，必须显式关闭）。
    try {
        $action = New-ScheduledTaskAction -Execute $exe -WorkingDirectory (Split-Path $exe)
        $trigger = New-ScheduledTaskTrigger -AtLogon
        $settings = New-ScheduledTaskSettingsSet `
            -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
            -ExecutionTimeLimit ([TimeSpan]::Zero) `
            -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
            -MultipleInstances IgnoreNew
        $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive
        Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
            -Settings $settings -Principal $principal -Force | Out-Null
        Say "[OK] 已注册开机自启：$taskName"
        Say "     触发：登录时    崩溃恢复：1 分钟后自动拉起"
        Say "     卸载：重新运行本脚本加 -Remove 参数"
    }
    catch {
        Say "[ERROR] 注册失败: $_"
    }
}

if ($LogFile) {
    Set-Content -Path $LogFile -Value ($output -join "`r`n") -Encoding UTF8
}
else {
    Read-Host "按回车退出"
}
