"""分析系统一键打包：PyInstaller onedir + 外置权重 + .env 模板。

用法（analysis_system 环境）:
    python tools/build_exe.py            # 首次自动装 pyinstaller
    python tools/build_exe.py --no-build # 只做后置步骤（复制权重/.env）

产物: dist/分析系统/（约 6-7GB，可整目录 zip 分发）
说明: 相机侧桥接 tools/shm_file_bridge.py 不在 exe 内——采集机仍需
      Python 环境运行该脚本（后续可按需打独立小 exe，不含 torch）。
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "分析系统"
WEIGHTS_SRC = [
    ROOT / "core" / "anomaly" / "dinomaly" / "weights" / "coating_anomaly_final.pth",
]
WEIGHTS_GLOB = (ROOT / "core" / "trend" / "weights", "*.ckpt")
ENV_TEMPLATE = """\
# 分析系统配置（改后重启生效）
# IPC_MODE: shm=共享内存实时链路（需相机侧桥接进程），file=文件轮询（默认）
IPC_MODE=file
# 异常警告弹窗自动关闭时长（毫秒），0 表示手动关闭
ANOMALY_ALERT_AUTO_CLOSE_MS=5000
# 异常警告弹窗开关：1=弹出（默认），0=不弹窗仅记日志
ANOMALY_ALERT_POPUP_ENABLED=1
"""


def ensure_pyinstaller() -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("[BUILD] installing pyinstaller ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])


def build() -> None:
    print("[BUILD] running pyinstaller (10-30 min for torch) ...")
    subprocess.check_call([
        sys.executable, "-m", "PyInstaller", "--noconfirm", "分析系统.spec",
    ], cwd=str(ROOT))


def fix_dll_conflicts() -> None:
    """清理 PyInstaller 收集的旧版运行时，统一走 System32。

    根因（WinError 1114 / c10.dll 初始化失败）：PyQt5 wheel 自带的
    MSVCP140/VCRUNTIME140 为 14.26（2020 年），经由 hook 注册的搜索
    目录抢先于 System32 被加载，torch 2.11 绑定其缺失的新导出即崩。
    删除后 Qt 回落 System32（14.50 向后兼容）。conda 的 UCRT stub
    （api-ms-win-*）与 ucrtbase.dll 同理一并清理，减重且防劫持。
    """
    internal = OUT / "_internal"
    qt_bin = internal / "PyQt5" / "Qt5" / "bin"
    removed = 0
    if qt_bin.is_dir():
        for name in ("MSVCP140.dll", "VCRUNTIME140.dll", "VCRUNTIME140_1.dll"):
            f = qt_bin / name
            if f.exists():
                f.unlink()
                removed += 1
    for f in internal.glob("api-ms-win-*.dll"):
        f.unlink()
        removed += 1
    ucrt = internal / "ucrtbase.dll"
    if ucrt.exists():
        ucrt.unlink()
        removed += 1
    print(f"[BUILD] removed {removed} conflicting runtime DLLs (PyQt5 CRT 14.26 / UCRT stubs)")


def post_copy() -> None:
    if not OUT.is_dir():
        sys.exit(f"[BUILD] 产物目录不存在: {OUT}（先完成构建）")
    fix_dll_conflicts()
    wdir = OUT / "weights"
    wdir.mkdir(exist_ok=True)
    print("[BUILD] copying external weights ...")
    for src in WEIGHTS_SRC:
        if src.is_file():
            shutil.copy2(src, wdir / src.name)
            print(f"  + {src.name} ({src.stat().st_size / 1e9:.2f} GB)")
        else:
            print(f"  ! 缺失（跳过）: {src}")
    pattern_dir, pattern = WEIGHTS_GLOB
    for src in sorted(pattern_dir.glob(pattern)):
        shutil.copy2(src, wdir / src.name)
        print(f"  + {src.name} ({src.stat().st_size / 1e6:.1f} MB)")
    env_path = OUT / ".env"
    if not env_path.exists():
        env_path.write_text(ENV_TEMPLATE, encoding="utf-8")
        print("[BUILD] wrote .env template")


def report() -> None:
    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"[BUILD] done. {OUT}  total {total / 1e9:.2f} GB")
    print(f"[BUILD] 冒烟验证: {OUT / '分析系统.exe'} --smoke"
          f"（结果见 {OUT / '_smoke_result.txt'}，退出码 0 为通过）")
    print("[BUILD] 手工验证: 双击 exe 跑单张检测 + 切 dinomaly 引擎 + shm 链路")


if __name__ == "__main__":
    ensure_pyinstaller()
    if "--no-build" not in sys.argv:
        build()
    post_copy()
    report()
