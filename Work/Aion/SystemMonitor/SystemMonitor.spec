# -*- mode: python ; coding: utf-8 -*-

import certifi
import ddddocr
from pathlib import Path
from PyInstaller.utils.hooks import collect_dynamic_libs
import ssl

# 收集 OpenSSL 依赖库
ssl_binaries = collect_dynamic_libs("ssl")

block_cipher = None

ddddocr_path = Path(ddddocr.__file__).parent
ddddocr_files = []
for file in ddddocr_path.glob('*.onnx'):
    ddddocr_files.append((str(file), 'ddddocr'))
for file in ddddocr_path.glob('*.proto'):
    ddddocr_files.append((str(file), 'ddddocr'))

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('config.json', '.'), 
        (certifi.where(), 'certifi'),
    ] + ddddocr_files,
    hiddenimports=['ddddocr', 'requests', 'PIL', 'schedule', 'email', 'smtplib', 'logging', 'ssl', 'cryptography',],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SystemMonitor',  # 直接在这里设置最终名称
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codepage='utf-8',
    icon=['monitor.ico']
)