# -*- mode: python ; coding: utf-8 -*-
"""分析系统 PyInstaller 打包配置（onedir / CUDA / 权重外置）。

构建：pyinstaller --noconfirm 分析系统.spec （或 tools/build_exe.bat）
策略：
- core/ 算法包整体作为 datas 复制进 _internal/core/（排除权重/_archived/
  __pycache__）：引擎 utils/dinomaly_engine.py、utils/trend_engine.py 的
  运行时 sys.path.insert 推导 Path(__file__).parent.parent 在 frozen 下
  指向 _internal/，算法包零改动可用
- 权重不进包：构建脚本后置复制到 dist/分析系统/weights/（外置可替换），
  入口 main_window._setup_frozen_environment 设 ANOMALY_MODEL_PATH/
  TREND_MODEL_PATH 指向它
- hiddenimports 为动态导入链的第三方库（静态分析盲区），缺失在 --smoke
  时以 ImportError 暴露，逐个补
"""
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

PROJECT = Path(SPECPATH)

# timm 整包收集：动态导入链引用且内置架构配置 json（datas）
timm_datas, timm_binaries, timm_hiddenimports = collect_all('timm')

# ---- core/ 数据文件：递归收集，排除权重（外置）、_archived、__pycache__ ----
EXCLUDE_DIRS = {"weights", "_archived", "__pycache__"}
core_datas = []
for src in sorted((PROJECT / "core").rglob("*")):
    if not src.is_file():
        continue
    rel = src.relative_to(PROJECT)
    if any(part in EXCLUDE_DIRS for part in rel.parts):
        continue
    core_datas.append((str(src), str(rel.parent)))

a = Analysis(
    ['main_window.py'],
    pathex=[str(PROJECT)],
    binaries=timm_binaries,
    datas=core_datas + timm_datas,
    hiddenimports=[
        # 动态导入链（models/dataloader/utils_ 等 core 包）引用的第三方库，
        # 按 core/*.py 的 import 语句精确收集（延迟导入静态分析不到）
        'torchinfo',
        'sklearn',
        'sklearn.utils._weight_vector',
        'sklearn.cluster',
        'sklearn.metrics',
        'skimage.measure',
        'scipy.interpolate',
        'scipy.ndimage',
        'pandas',
        'PIL',
    ] + timm_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'IPython',
        'pytest',
        'tests',
        'tools',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='分析系统',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # torch/cv2 大 DLL 经 UPX 会损坏或启动变慢，禁用
    console=False,  # GUI 应用；--smoke 输出写 _smoke_result.txt
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='分析系统',
)
