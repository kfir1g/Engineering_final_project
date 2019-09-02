# -*- mode: python ; coding: utf-8 -*-

import sys
sys.setrecursionlimit(5000)
block_cipher = None
from PyInstaller.utils.hooks import collect_submodules


a = Analysis(['GUI.py'],
             pathex=['C:\\Users\\Kfir\\Documents\\Programming\\Scaphoid-project'],
             binaries=[],
             datas=[],
             hiddenimports=collect_submodules('sklearn'),
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='GUI',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False)
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='GUI')
