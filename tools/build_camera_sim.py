"""相机端模拟器一键打包：PyInstaller onedir（不含 torch，产物 ~200MB 级）。

用法（analysis_system 环境）:
    python tools/build_camera_sim.py

产物: dist/相机端模拟器/（相机端模拟器.exe + _internal/，可整目录分发）
冒烟: dist/相机端模拟器/相机端模拟器.exe --smoke
      （输出写 exe 旁 _smoke_result.txt，退出码 0 为通过）
说明: 与主程序 tools/build_exe.py 相互独立、可分别运行；本工具不含
      torch/权重，采集机可直接用 exe，无需 Python 环境。
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "相机端模拟器"
ENV_PRESERVED = ROOT / "dist" / ".camera_sim_env.preserved"  # COLLECT 只清 相机端模拟器 子目录
ENV_TEMPLATE = """\
# 相机端模拟器配置（改后重启生效）
# 共享内存命名空间（须与相机端接入指南/分析端一致，默认 Local\\IAP）
#SHM_NAMESPACE=Local\\IAP
"""


def ensure_pyinstaller() -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("[BUILD] installing pyinstaller ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])


def preserve_user_env() -> None:
    """COLLECT 重建 dist/相机端模拟器 前备份用户自定义 .env。"""
    src = OUT / ".env"
    if src.is_file():
        shutil.copy2(src, ENV_PRESERVED)
        print("[BUILD] preserved user .env -> dist/.camera_sim_env.preserved")


def restore_user_env() -> None:
    """构建后恢复用户 .env；无备份且 .env 缺失时才写模板。"""
    env_path = OUT / ".env"
    if ENV_PRESERVED.is_file():
        shutil.copy2(ENV_PRESERVED, env_path)
        ENV_PRESERVED.unlink()
        print("[BUILD] restored user .env (customizations kept)")
    elif not env_path.exists():
        env_path.write_text(ENV_TEMPLATE, encoding="utf-8")
        print("[BUILD] wrote .env template")


def build() -> None:
    print("[BUILD] running pyinstaller (a few minutes, no torch) ...")
    subprocess.check_call([
        sys.executable, "-m", "PyInstaller", "--noconfirm",
        "--windowed", "--name", "相机端模拟器",
        "--paths", ".",
        str(ROOT / "tools" / "camera_simulator.py"),
    ], cwd=str(ROOT))


def fix_dll_conflicts() -> None:
    """清理 PyQt5 wheel 自带的旧版 CRT（MSVCP140/VCRUNTIME140 14.26）与
    conda UCRT stub，统一回落 System32——与 tools/build_exe.py 同款清理，
    减重且防 DLL 劫持（本工具无 torch，风险更低，清理保持一致）。"""
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


def report() -> None:
    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"[BUILD] done. {OUT}  total {total / 1e6:.0f} MB")
    print("[BUILD] 冒烟验证: 相机端模拟器.exe --smoke"
          f"（结果见 {OUT / '_smoke_result.txt'}，退出码 0 为通过）")


if __name__ == "__main__":
    ensure_pyinstaller()
    preserve_user_env()
    build()
    fix_dll_conflicts()
    restore_user_env()
    report()
