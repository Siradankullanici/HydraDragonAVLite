#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
from datetime import datetime, timedelta
import time

main_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(main_dir)
sys.path.insert(0, main_dir)

# Set script directory
script_dir = os.getcwd()

# Define log directories and files
log_directory = os.path.join(script_dir, "log")
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# Separate log files for different purposes
stdout_console_log_file = os.path.join(
    log_directory, "antivirusconsolestdout.log"
)
stderr_console_log_file = os.path.join(
    log_directory, "antivirusconsolestderr.log"
)
application_log_file = os.path.join(
    log_directory, "antivirus.log"
)

# Configure logging for application log
logging.basicConfig(
    filename=application_log_file,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# Redirect stdout to stdout console log
sys.stdout = open(
    stdout_console_log_file, "w", encoding="utf-8", errors="ignore"
)

# Redirect stderr to stderr console log
sys.stderr = open(
    stderr_console_log_file, "w", encoding="utf-8", errors="ignore"
)

# Logging for application initialization
logging.info(
    "Application started at %s",
    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
)

# Start timing total duration
total_start_time = time.time()

# Measure and logging.info time taken for each import
start_time = time.time()
import hashlib
logging.info(f"hashlib module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import io
logging.info(f"io module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import webbrowser
logging.info(f"webbrowser module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
from uuid import uuid4 as uniquename
logging.info(f"uuid.uuid4.uniquename loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import shutil
logging.info(f"shutil module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import subprocess
logging.info(f"subprocess module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import threading
logging.info(f"threading module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
from concurrent.futures import ThreadPoolExecutor
logging.info(f"concurrent.futures.ThreadPoolExecutor module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import re
logging.info(f"re module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import json
logging.info(f"json module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QTextEdit, QFileDialog,
                               QFrame, QStackedWidget,
                               QApplication, QButtonGroup, QGroupBox)
logging.info(f"PySide6.QtWidgets modules loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
from PySide6.QtCore import (Qt, QPropertyAnimation, QEasingCurve, QThread,
                            Signal, QPoint, QParallelAnimationGroup, Property, QRect)
logging.info(f"PySide6.QtCore modules loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
from PySide6.QtGui import (QColor, QPainter, QBrush, QLinearGradient, QPen,
                           QPainterPath, QRadialGradient, QIcon, QPixmap)
logging.info(f"PySide6.QtGui.QIcon module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import pefile
logging.info(f"pefile module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import pyzipper
logging.info(f"pyzipper module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import tarfile
logging.info(f"tarfile module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import yara
logging.info(f"yara module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import yara_x
logging.info(f"yara_x module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import psutil
logging.info(f"psutil module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
from notifypy import Notify
logging.info(f"notifypy.Notify module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
from watchdog.observers import Observer
logging.info(f"watchdog.observers.Observer module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
from watchdog.events import FileSystemEventHandler
logging.info(f"watchdog.events.FileSystemEventHandler module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import win32file
logging.info(f"win32file module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import win32con
logging.info(f"win32con module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import wmi
logging.info(f"wmi module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import numpy as np
logging.info(f"numpy module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()

from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from scapy.layers.dns import DNS, DNSQR, DNSRR
from scapy.sendrecv import sniff

logging.info(f"scapy modules loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import ast
logging.info(f"ast module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import ctypes
logging.info(f"ctypes module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
from ctypes import wintypes
logging.info(f"ctypes.wintypes module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
from ctypes import byref
logging.info(f"ctypes.byref module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import comtypes
logging.info(f"comtypes module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
from comtypes.automation import VARIANT
logging.info(f"comtypes.automation.VARIANT module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
from comtypes import CoInitialize
logging.info(f"comtypes.CoInitialize module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
from comtypes.client import CreateObject, GetModule
logging.info(f"comtypes.client.CreateObject and GetModule modules loaded in {time.time() - start_time:.6f} seconds")

# Generate the oleacc module
start_time = time.time()
GetModule('oleacc.dll')
from comtypes.gen import Accessibility  # Usually oleacc maps to this
logging.info(f"comtypes.gen.Accessibility module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import ipaddress
logging.info(f"ipaddress module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
from urllib.parse import urlparse
logging.info(f"urllib.parse.urlparse module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import spacy
logging.info(f"spacy module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import csv
logging.info(f"csv module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import struct
logging.info(f"struct module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
from importlib.util import MAGIC_NUMBER
logging.info(f"importlib.util.MAGIC_NUMBER module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import string
logging.info(f"string module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import chardet
logging.info(f"chardet module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import difflib
logging.info(f"difflib module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import zlib
logging.info(f"zlib module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import marshal
logging.info(f"marshal module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import base64
logging.info(f"base64 module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import base32_crockford
logging.info(f"base32_crockford module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import binascii
logging.info(f"binascii module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
from transformers import AutoTokenizer, AutoModelForCausalLM
logging.info(f"transformers.AutoTokenizer and AutoModelForCausalLM modules loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
from accelerate import Accelerator
logging.info(f"accelerate.Accelerator module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import py7zr
logging.info(f"py7zr module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import pymem
logging.info(f"pymem module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import inspect
logging.info(f"inspect module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import zstandard
logging.info(f"zstandard module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
from typing import Optional, Tuple, BinaryIO, Dict, Any, List, Set
logging.info(f"typing, Optional, Tuple, BinaryIO, Dict and Any module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import types
logging.info(f"types module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
logging.info(f"cryptography.hazmat.primitives.ciphers, Cipher, algorithms, modes module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import debloat.processor
logging.info(f"debloat modules loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
from Crypto.Cipher import AES
logging.info(f"Crpyto.Cipher.AES module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
from Crypto.Util import Counter
logging.info(f"Crpyto.Cipher.Counter module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
from pathlib import Path, WindowsPath
logging.info(f"pathlib.Path module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
import requests
logging.info(f"requests module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
from functools import wraps
logging.info("functoools.wraps module loaded in {time.time() - start_time:.6f} seconds")

start_time = time.time()
from xdis.unmarshal import load_code
logging.info("xdis.unmarshal.load_code module loaded in {time.time() - start_time:.6f} seconds")

# Calculate and logging.info total time
total_end_time = time.time()
total_duration = total_end_time - total_start_time
logging.info(f"Total time for all imports: {total_duration:.6f} seconds")

# Initialize the accelerator and device
accelerator = Accelerator()
device = accelerator.device

# get the full path to the currently running Python interpreter
python_path = sys.executable

# Define the paths
reports_dir = os.path.join(script_dir, "reports")
scan_report_path = os.path.join(reports_dir, "scan_report.json")
enigma_extracted_dir = os.path.join(script_dir, "enigma_extracted")
inno_unpack_dir = os.path.join(script_dir, "innounp-2")
upx_dir = os.path.join(script_dir, "upx-5.0.1-win64")
upx_path = os.path.join(upx_dir, "upx.exe")
upx_extracted_dir = os.path.join(script_dir, "upx_extracted_dir")
inno_unpack_path = os.path.join(inno_unpack_dir, "innounp.exe")
inno_setup_unpacked_dir = os.path.join(script_dir, "inno_setup_unpacked")
decompiled_dir = os.path.join(script_dir, "decompiled")
known_extensions_dir = os.path.join(script_dir, "known_extensions")
FernFlower_path = os.path.join(jar_decompiler_dir, "fernflower.jar")
system_file_names_path = os.path.join(known_extensions_dir, "systemfilenames.txt")
extensions_path = os.path.join(known_extensions_dir, "extensions.txt")
detectiteasy_dir = os.path.join(script_dir, "detectiteasy")
deteciteasy_plain_text_dir = os.path.join(script_dir, "deteciteasy_plain_text")
detectiteasy_console_path = os.path.join(detectiteasy_dir, "diec.exe")
machine_learning_dir = os.path.join(script_dir, "machinelearning")
machine_learning_results_json = os.path.join(machine_learning_dir, "results.json")
yara_dir = os.path.join(script_dir, "yara")
excluded_rules_dir = os.path.join(script_dir, "excluded")
excluded_rules_path = os.path.join(excluded_rules_dir, "excluded_rules.txt")
antivirus_list_path = os.path.join(script_dir, "hosts", "antivirus_list.txt")
yaraxtr_yrc_path = os.path.join(yara_dir, "yaraxtr.yrc")
cx_freeze_yrc_path = os.path.join(yara_dir, "cx_freeze.yrc")
compiled_rule_path = os.path.join(yara_dir, "compiled_rule.yrc")
yarGen_rule_path = os.path.join(yara_dir, "machinelearning.yrc")
icewater_rule_path = os.path.join(yara_dir, "icewater.yrc")
valhalla_rule_path = os.path.join(yara_dir, "valhalla-rules.yrc")
Open_Hydra_Dragon_Anti_Rootkit_path = os.path.join(script_dir, "OpenHydraDragonAntiRootkit.py")

antivirus_domains_data = []

# Resolve system drive path
system_drive = os.getenv("SystemDrive", "C:") + os.sep
# Resolve Program Files directory via environment (fallback to standard path)
program_files = os.getenv("ProgramFiles", os.path.join(system_drive, "Program Files"))
# Get SystemRoot (usually C:\Windows)
system_root = os.getenv("SystemRoot", os.path.join(system_drive, "Windows"))
# Fallback to %SystemRoot%\System32 if %System32% is not set
system32_dir = os.getenv("System32", os.path.join(system_root, "System32"))

# Snort base folder path
snort_folder = os.path.join(system_drive, "Snort")

# File paths and configurations
log_folder = os.path.join(snort_folder, "log")
log_path = os.path.join(log_folder, "alert.ids")
snort_config_path = os.path.join(snort_folder, "etc", "snort.conf")
snort_exe_path = os.path.join(snort_folder, "bin", "snort.exe")
sandboxie_dir = os.path.join(program_files, "Sandboxie")
sandboxie_path = os.path.join(sandboxie_dir, "Start.exe")
sandboxie_control_path = os.path.join(sandboxie_dir, "SbieCtrl.exe")
device_args = [f"-i {i}" for i in range(1, 26)]  # Fixed device arguments
username = os.getlogin()
sandboxie_folder = os.path.join(system_drive, "Sandbox", username, "DefaultBox")
main_drive_path = os.path.join(sandboxie_folder, "drive", system_drive.strip(":"))

def get_sandbox_path(original_path: str | Path) -> Path:
    original_path = Path(original_path)
    sandboxie_folder_path = Path(sandboxie_folder)

    drive_letter = original_path.drive.rstrip(":")  # e.g., "C"
    rest_path = original_path.relative_to(original_path.anchor).parts

    sandbox_path = sandboxie_folder_path / "drive" / drive_letter / Path(*rest_path)
    return sandbox_path

# Derived sandbox system root path
sandbox_system_root_directory = get_sandbox_path(system_root)

# Derived sandbox system32 path
sandbox_system32_directory = get_sandbox_path(system32_dir)

# Derived sandbox scan report path
sandbox_scan_report_path = get_sandbox_path(scan_report_path)

# Constant special item ID list value for desktop folder
CSIDL_DESKTOPDIRECTORY = 0x0010

# Flag for SHGetFolderPath
SHGFP_TYPE_CURRENT = 0

# Convenient shorthand for this function
SHGetFolderPathW = ctypes.windll.shell32.SHGetFolderPathW


def _get_folder_path(csidl):
    """Get the path of a folder identified by a CSIDL value."""
    # Create a buffer to hold the return value from SHGetFolderPathW
    buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)

    # Return the path as a string
    SHGetFolderPathW(None, csidl, None, SHGFP_TYPE_CURRENT, buf)
    return str(buf.value)


def get_desktop():
    """Return the current user's Desktop folder."""
    return _get_folder_path(CSIDL_DESKTOPDIRECTORY)

def get_sandboxie_log_folder():
    """Return the sandboxie log folder path on the desktop."""
    return f'{get_desktop()}\\DONTREMOVEHydraDragonAntivirusLogs'

ntdll_path = os.path.join(system32_dir, "ntdll.dll")
sandboxed_ntdll_path = os.path.join(sandbox_system32_directory, "ntdll.dll")
drivers_path = os.path.join(system32_dir, "drivers")
hosts_path = f'{drivers_path}\\hosts'
HydraDragonAntivirus_sandboxie_path = get_sandbox_path(script_dir)
sandboxie_log_folder = get_sandboxie_log_folder()
homepage_change_path = f'{sandboxie_log_folder}\\DONTREMOVEHomePageChange.txt'
HiJackThis_log_path = f'{HydraDragonAntivirus_sandboxie_path}\\HiJackThis\\HiJackThis.log'
de4dot_sandboxie_dir = f'{HydraDragonAntivirus_sandboxie_path}\\de4dot_extracted_dir'
python_deobfuscated_sandboxie_dir = f'{HydraDragonAntivirus_sandboxie_path}\\python_deobfuscated'
version_flag = f"-{sys.version_info.major}.{sys.version_info.minor}"

script_exts = {
    '.vbs', '.vbe', '.js', '.jse', '.bat', '.url',
    '.cmd', '.hta', '.ps1', '.psm1', '.wsf', '.wsb', '.sct'
}

# Known Enigma versions -> working evbunpack flags
PACKER_FLAGS = {
    "11.00": ["-pe", "10_70"],
    "10.70": ["-pe", "10_70"],
    "9.70":  ["-pe", "9_70"],
    "7.80":  ["-pe", "7_80", "--legacy-fs"],
}

# Define the list of known rootkit filenames
known_rootkit_files = [
    'MoriyaStreamWatchmen.sys',
    # Add more rootkit filenames here if needed
]

uefi_100kb_paths = [
    rf'{sandboxie_folder}\drive\X\EFI\Microsoft\Boot\SecureBootRecovery.efi'
]

uefi_paths = [
    rf'{sandboxie_folder}\drive\X\EFI\Microsoft\Boot\bootmgfw.efi',
    rf'{sandboxie_folder}\drive\X\EFI\Microsoft\Boot\bootmgr.efi',
    rf'{sandboxie_folder}\drive\X\EFI\Microsoft\Boot\memtest.efi',
    rf'{sandboxie_folder}\drive\X\EFI\Boot\bootx64.efi'
]
snort_command = [snort_exe_path] + device_args + ["-c", snort_config_path, "-A", "fast"]

# Custom flags for directory changes
FILE_NOTIFY_CHANGE_LAST_ACCESS = 0x00000020
FILE_NOTIFY_CHANGE_CREATION = 0x00000040
FILE_NOTIFY_CHANGE_EA = 0x00000080
FILE_NOTIFY_CHANGE_STREAM_NAME = 0x00000200
FILE_NOTIFY_CHANGE_STREAM_SIZE = 0x00000400
FILE_NOTIFY_CHANGE_STREAM_WRITE = 0x00000800

directories_to_scan = [pd64_extracted_dir, enigma_extracted_dir, sandboxie_folder, copied_sandbox_and_main_files_dir, decompiled_dir, inno_setup_unpacked_dir, FernFlower_decompiled_dir, jar_extracted_dir, nuitka_dir, dotnet_dir, obfuscar_dir, de4dot_extracted_dir, pyinstaller_extracted_dir, cx_freeze_extracted_dir, commandlineandmessage_dir, pe_extracted_dir, zip_extracted_dir, tar_extracted_dir, seven_zip_extracted_dir, general_extracted_with_7z_dir, nuitka_extracted_dir, advanced_installer_extracted_dir, processed_dir, python_source_code_dir, pylingual_extracted_dir, python_deobfuscated_dir, python_deobfuscated_marshal_pyc_dir, pycdas_extracted_dir, nuitka_source_code_dir, memory_dir, debloat_dir, resource_extractor_dir, ungarbler_dir, ungarbler_string_dir, html_extracted_dir]

# ClamAV base folder path
clamav_folder = os.path.join(program_files, "ClamAV")

# 7-Zip base folder path
seven_zip_folder = os.path.join(program_files, "7-Zip")

# ClamAV file paths and configurations
clamdscan_path = os.path.join(clamav_folder, "clamdscan.exe")
freshclam_path = os.path.join(clamav_folder, "freshclam.exe")
clamav_database_directory_path = os.path.join(clamav_folder, "database")
clamav_file_paths = [
    os.path.join(clamav_database_directory_path, "daily.cvd"),
    os.path.join(clamav_database_directory_path, "daily.cld")
]

for make_directory in MANAGED_DIRECTORIES:
  if not os.path.exists(make_directory):
    os.makedirs(make_directory)

# Sandboxie folders
os.makedirs(sandboxie_folder, exist_ok=True)
os.makedirs(sandbox_system_root_directory, exist_ok=True)

# Counter for ransomware detection
ransomware_detection_count = 0

def reset_flags():
    global main_file_path, pyinstaller_archive, full_python_version, pyz_version_match
    main_file_path = None
    pyinstaller_archive = None
    full_python_version = None
    pyz_version_match = False
reset_flags()

# Cache of { file_path: last_md5 }
file_md5_cache: dict[str, str] = {}

# Global cache: md5 -> (die_output, plain_text_flag)
die_cache: Dict[str, Tuple[str, bool]] = {}

# Separate cache for "binary-only" DIE results
binary_die_cache: Dict[str, str] = {}

def compute_md5_via_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def compute_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def try_unpack_enigma1(input_exe: str) -> str | None:
    """
    Attempts to unpack an Enigma protected EXE by trying each known
    version+flag combo until one succeeds.

    :param input_exe: Path to the Enigma protected executable.
    :return: Path to the directory where files were extracted, or
             None if all attempts failed.
    """
    exe_name = Path(input_exe).stem

    for version, flags in PACKER_FLAGS.items():
        # Create a subdir for this version attempt: <exe_name>_v<version>
        version_dir = os.path.join(enigma_extracted_dir, f"{exe_name}_v{version}")
        os.makedirs(version_dir, exist_ok=True)

        cmd = ["evbunpack"] + flags + [input_exe, version_dir]
        logging.info(f"Trying Enigma protected v{version} flags: {flags}")
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        if proc.returncode == 0:
            logging.info(f"Successfully unpacked with version {version} into {version_dir}")
            return version_dir

        logging.warning(
            f"Attempt v{version} failed (exit {proc.returncode}). Output:\n{proc.stdout}"
        )

    logging.error(
        f"All unpack attempts failed for {input_exe}. Tried versions: {', '.join(PACKER_FLAGS)}"
    )
    return None

def is_plain_text(data: bytes,
                  null_byte_threshold: float = 0.01,
                  printable_threshold: float = 0.95) -> bool:
    """
    Heuristic: data is plain text if
      1. It contains very few null bytes,
      2. A high fraction of bytes are printable or common whitespace,
      3. And it decodes cleanly in some text encoding (e.g. UTF-8, Latin-1).

    :param data:       raw file bytes
    :param null_byte_threshold:
                       max fraction of bytes that can be zero (0x00)
    :param printable_threshold:
                       min fraction of bytes in printable + whitespace set
    """
    if not data:
        return True

    # 1) Null byte check
    nulls = data.count(0)
    if nulls / len(data) > null_byte_threshold:
        return False

    # 2) Printable char check
    printable = set(bytes(string.printable, 'ascii'))
    count_printable = sum(b in printable for b in data)
    if count_printable / len(data) < printable_threshold:
        return False

    # 3) Try a text decoding
    #    Use chardet to guess encoding
    guess = chardet.detect(data)
    enc = guess.get('encoding') or 'utf-8'
    try:
        data.decode(enc)
        return True
    except (UnicodeDecodeError, LookupError):
        return False

def is_plain_text_data(die_output):
    """
    Checks if the DIE output does indicate plain text, suggesting it is plain text data.
    """
    if die_output and "Format: plain text" in die_output.lower():
        logging.info("DIE output does not contain plain text; identified as non-plain text data.")
        return True
    return False

def is_valid_ip(ip_string: str) -> bool:
    """
    Returns True if ip_string is a valid public IPv4 or IPv6 address,
    False otherwise. Logs details about invalid cases.
    """

    # --- strip off port if present ---
    original = ip_string
    # IPv6 with brackets, e.g. "[2001:db8::1]:443"
    if ip_string.startswith('[') and ']' in ip_string:
        ip_core, sep, port = ip_string.partition(']')
        if sep and port.startswith(':') and port[1:].isdigit():
            ip_string = ip_core.lstrip('[')
            logging.debug(f"Stripped port from bracketed IPv6: {original!r} {ip_string!r}")
    # IPv4 or unbracketed IPv6: split on last colon only if it looks like a port
    elif ip_string.count(':') == 1:
        ip_part, port = ip_string.rsplit(':', 1)
        if port.isdigit():
            ip_string = ip_part
            logging.debug(f"Stripped port from IPv4/unbracketed: {original!r} {ip_string!r}")
    # else: leave IPv6 with multiple colons intact

    logging.info(f"Validating IP: {ip_string!r}")
    try:
        ip_obj = ipaddress.ip_address(ip_string)
        logging.debug(f"Parsed IP object: {ip_obj} (version {ip_obj.version})")
    except ValueError:
        logging.error(f"Invalid IP syntax: {ip_string!r}")
        return False

    # exclusion categories
    if ip_obj.is_private:
        logging.info(f"Excluded private IP: {ip_obj}")
        return False
    if ip_obj.is_loopback:
        logging.info(f"Excluded loopback IP: {ip_obj}")
        return False
    if ip_obj.is_link_local:
        logging.info(f"Excluded link-local IP: {ip_obj}")
        return False
    if ip_obj.is_multicast:
        logging.info(f"Excluded multicast IP: {ip_obj}")
        return False
    if ip_obj.is_reserved:
        logging.info(f"Excluded reserved IP: {ip_obj}")
        return False

    # valid public IP
    logging.info(f"Valid public IPv{ip_obj.version} address: {ip_obj}")
    return True

def sanitize_filename(filename: str) -> str:
    """
    Sanitize the filename by replacing invalid characters for Windows.
    """
    return filename.replace(':', '_').replace('\\', '_').replace('/', '_')

def ublock_detect(url):
    """
    Check if the given URL should be detected by the uBlock-style rule.

    The rule matches:
      - URLs that fit the regex pattern.
      - Only applies to main document requests.

    The exception: if the URL includes 'steamcommunity.com', then the rule is not applied.
    """
    # First, check if the URL matches the regex pattern.
    if not UBLOCK_REGEX.match(url):
        return False

    # Apply exception: if the URL's domain includes "steamcommunity.com", ignore it.
    if 'steamcommunity.com' in url:
        return False

    return True

def get_resource_name(entry):
    # Get the resource name, which might be a string or an ID
    if hasattr(entry, 'name') and entry.name is not None:
        return str(entry.name)
    else:
        return str(entry.id)

# Read the file types from extensions.txt with try-except
fileTypes = []
try:
    if os.path.exists(extensions_path):
        with open(extensions_path, 'r') as ext_file:
            fileTypes = [line.strip() for line in ext_file.readlines()]
except Exception as ex:
    logging.info(f"Error reading {extensions_path}: {ex}")

logging.info(f"File types read from {extensions_path}: {fileTypes}")

# Read antivirus process list from antivirusprocesslist.txt with try-except.
antivirus_process_list = []
try:
    if os.path.exists(antivirus_process_list_path):
        with open(antivirus_process_list_path, 'r') as av_file:
            antivirus_process_list = [line.strip() for line in av_file if line.strip()]
except Exception as ex:
    logging.info(f"Error reading {antivirus_process_list_path}: {ex}")

logging.info(f"Antivirus process list read from {antivirus_process_list_path}: {antivirus_process_list}")

pe_file_paths = []  # List to store the PE file paths

# Initialize an empty dictionary for magic_bytes
magic_bytes = {}

try:
    # Read the magicbytes.txt file and populate the dictionary
    with open(magic_bytes_path, "r") as file:
        for line in file:
            # Split each line into magic bytes and file type
            parts = line.strip().split(": ")
            if len(parts) == 2:
                magic, file_type = parts
                magic_bytes[magic] = file_type

    # If reading and processing is successful, logging.info the dictionary
    logging.info("Magic bytes have been successfully loaded.")

except FileNotFoundError:
    logging.error(f"Error: The file {magic_bytes_path} was not found.")
except Exception as e:
    logging.error(f"An error occurred: {e}")

def get_unique_output_path(output_dir: Path, base_name) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(base_name)  # <- convert here

    stem = sanitize_filename(base_name.stem)
    suffix = base_name.suffix

    timestamp = int(time.time())
    candidate = output_dir / f"{stem}_{timestamp}{suffix}"

    if candidate.exists():
        counter = 1
        while True:
            candidate = output_dir / f"{stem}_{timestamp}_{counter}{suffix}"
            if not candidate.exists():
                break
            counter += 1

    return candidate

def analyze_file_with_die(file_path):
    """
    Runs Detect It Easy (DIE) on the given file once and returns the DIE output (plain text).
    The output is also saved to a unique .txt file.
    """
    try:
        logging.info(f"Analyzing file: {file_path} using Detect It Easy...")
        output_dir = Path(deteciteasy_plain_text_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define the base name for the output text file
        base_name = Path(file_path).with_suffix(".txt")
        txt_output_path = get_unique_output_path(output_dir, base_name)

        # Run the DIE command once with the -p flag for plain output
        result = subprocess.run(
            [detectiteasy_console_path, "-p", file_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore"
        )

        # Save the plain text output
        with open(txt_output_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(result.stdout)

        logging.info(f"Analysis result saved to {txt_output_path}")
        return result.stdout

    except subprocess.SubprocessError as ex:
        logging.error(
            f"Error in {inspect.currentframe().f_code.co_name} while running Detect It Easy for {file_path}: {ex}"
        )
        return None
    except Exception as ex:
        logging.error(
            f"General error in {inspect.currentframe().f_code.co_name} while running Detect It Easy for {file_path}: {ex}"
        )
        return None

def get_die_output(path: str) -> Tuple[str, bool]:
    """
    Returns (die_output, plain_text_flag), caching results by content MD5.
    """
    file_md5 = compute_md5(path)
    if file_md5 in die_cache:
        return die_cache[file_md5]

    # first time for this content:
    with open(path, "rb") as f:
        peek = f.read(8192)
    if is_plain_text(peek):
        die_output = "Binary\n    Format: plain text"
        plain_text_flag = True
    else:
        die_output = analyze_file_with_die(path)
        plain_text_flag = is_plain_text_data(die_output)

    die_cache[file_md5] = (die_output, plain_text_flag)
    return die_output, plain_text_flag

def get_die_output_binary(path: str) -> str:
    """
    Returns die_output for a non plain text file, caching by content MD5.
    (Assumes the file isn't plain text, so always calls analyze_file_with_die()
     on cache miss.)
    """
    file_md5 = compute_md5(path)
    if file_md5 in binary_die_cache:
        return binary_die_cache[file_md5]

    # First time for this content: run DIE and cache
    die_output = analyze_file_with_die(path)
    binary_die_cache[file_md5] = die_output
    return die_output

def is_pe_file_from_output(die_output):
    """Checks if DIE output indicates a PE (Portable Executable) file."""
    if die_output and ("PE32" in die_output or "PE64" in die_output):
        logging.info("DIE output indicates a PE file.")
        return True
    logging.info(f"DIE output does not indicate a PE file: {die_output}")
    return False

def is_file_fully_unknown(die_output: str) -> bool:
    """
    Determines whether DIE output indicates an unrecognized binary file,
    ignoring any trailing error messages or extra lines.

    Returns True if the first two non-empty, whitespace-stripped lines are:
        Binary
        Unknown: Unknown
    """
    if not die_output:
        logging.info("No DIE output provided.")
        return False

    # Normalize: split into lines, strip whitespace, drop empty lines
    lines = [line.strip() for line in die_output.splitlines() if line.strip()]

    # We only care about the first two markers; ignore anything after.
    if len(lines) >= 2 and lines[0] == "Binary" and lines[1] == "Unknown: Unknown":
        logging.info("DIE output indicates an unknown file (ignoring extra errors).")
        return True
    else:
        logging.info(f"DIE output does not indicate an unknown file: {die_output!r}")
        return False

def debloat_pe_file(file_path):
    """
    Runs debloat.processor.process_pe on a PE file, writing all
    output into its own uniquely-named subdirectory of debloat_dir.
    """
    try:
        logging.info(f"Debloating PE file {file_path} for faster scanning.")

        # Flag for last-ditch processing
        last_ditch_processing = False

        # Normalize paths
        file_path = Path(file_path)
        base_dir  = Path(debloat_dir)

        # Build a unique output directory: debloat_dir/<stem>_<n>
        output_dir = base_dir / file_path.stem
        suffix = 1
        while output_dir.exists():
            output_dir = base_dir / f"{file_path.stem}_{suffix}"
            suffix += 1
        output_dir.mkdir(parents=True)

        # Load the PE into memory
        pe_data = file_path.read_bytes()
        pe      = pefile.PE(data=pe_data, fast_load=True)

        # Wrap logging.info so it accepts and ignores an 'end' kwarg
        def log_message(msg, *args, **kwargs):
            kwargs.pop('end', None)      # drop any 'end' argument
            logging.info(msg, *args, **kwargs)

        # Debloat into our new directory
        debloat.processor.process_pe(
            pe,
            log_message=log_message,
            last_ditch_processing=last_ditch_processing,
            out_path=str(output_dir),   # pass the folder path
            cert_preservation=True
        )

        # Verify that something landed in there
        if any(output_dir.iterdir()):
            logging.info(f"Debloated file(s) saved in: {output_dir}")
            return str(output_dir)
        else:
            logging.error(f"Debloating failed for {file_path}; {output_dir} is empty.")
            return None

    except Exception as ex:
        logging.error("Error during debloating of %s: %s", file_path, ex)

def process_file_data(file_path, die_output):
    """Process file data by decoding, removing magic bytes, and emitting a reversed lines version, saving outputs with .txt extension."""
    try:
        with open(file_path, 'rb') as data_file:
            data_content = data_file.read()

        # Peel off Base64/Base32 layers
        while True:
            # Base-64 first
            if isinstance(data_content, (bytes, bytearray)) and is_base64(data_content):
                decoded = decode_base64(data_content)
                if decoded is not None:
                    logging.info("Base64 layer removed.")
                    data_content = decoded
                    continue

            # then Base-32
            if isinstance(data_content, (bytes, bytearray)) and is_base32(data_content):
                decoded = decode_base32(data_content)
                if decoded is not None:
                    logging.info("Base32 layer removed.")
                    data_content = decoded
                    continue

            logging.info("No more base64 or base32 encoded data found.")
            break

        # strip out your magic bytes
        processed_data = remove_magic_bytes(data_content, die_output)

        # write the normal processed output with .txt extension
        base_name = os.path.basename(file_path)
        output_file_path = os.path.join(
            processed_dir,
            f'processed_{base_name}.txt'
        )
        with open(output_file_path, 'wb') as processed_file:
            processed_file.write(processed_data)
        logging.info(f"Processed data from {file_path} saved to {output_file_path}")

        # now create a reversed lines variant with .txt extension
        lines = processed_data.splitlines(keepends=True)
        reversed_lines_data = b''.join(lines[::-1])

        reversed_output_path = os.path.join(
            processed_dir,
            f'processed_reversed_lines_{base_name}.txt'
        )
        with open(reversed_output_path, 'wb') as rev_file:
            rev_file.write(reversed_lines_data)
        logging.info(f"Reversed lines data from {file_path} saved to {reversed_output_path}")

    except Exception as ex:
        logging.error(f"Error processing file {file_path}: {ex}")

def extract_infos(file_path, rank=None):
    """Extract information about file"""
    file_name = os.path.basename(file_path)
    if rank is not None:
        return {'file_name': file_name, 'numeric_tag': rank}
    else:
        return {'file_name': file_name}

def calculate_entropy(data: list) -> float:
    """Calculate Shannon entropy of data (provided as a list of integers)."""
    if not data:
        return 0.0

    total_items = len(data)
    value_counts = [data.count(i) for i in range(256)]  # Count occurrences of each byte (0-255)

    entropy = 0.0
    for count in value_counts:
        if count > 0:
            p_x = count / total_items
            entropy -= p_x * np.log2(p_x)

    return entropy

def get_callback_addresses(pe: pefile.PE, address_of_callbacks: int) -> List[int]:
    """Retrieve callback addresses from the TLS directory."""
    try:
        callback_addresses = []
        # Read callback addresses from the memory-mapped file
        while True:
            callback_address = pe.get_dword_at_rva(address_of_callbacks - pe.OPTIONAL_HEADER.ImageBase)
            if callback_address == 0:
                break  # End of callback list
            callback_addresses.append(callback_address)
            address_of_callbacks += 4  # Move to the next address (4 bytes for DWORD)

        return callback_addresses
    except Exception as e:
        logging.error(f"Error retrieving TLS callback addresses: {e}")
        return []

def analyze_tls_callbacks(pe: pefile.PE) -> Dict[str, Any]:
    """Analyze TLS (Thread Local Storage) callbacks and extract relevant details."""
    try:
        tls_callbacks = {}
        # Check if the PE file has a TLS directory
        if hasattr(pe, 'DIRECTORY_ENTRY_TLS'):
            tls = pe.DIRECTORY_ENTRY_TLS.struct
            tls_callbacks = {
                'start_address_raw_data': tls.StartAddressOfRawData,
                'end_address_raw_data': tls.EndAddressOfRawData,
                'address_of_index': tls.AddressOfIndex,
                'address_of_callbacks': tls.AddressOfCallBacks,
                'size_of_zero_fill': tls.SizeOfZeroFill,
                'characteristics': tls.Characteristics,
                'callbacks': []
            }

            # If there are callbacks, extract their addresses
            if tls.AddressOfCallBacks:
                callback_array = get_callback_addresses(pe, tls.AddressOfCallBacks)
                if callback_array:
                    tls_callbacks['callbacks'] = callback_array

        return tls_callbacks
    except Exception as e:
        logging.error(f"Error analyzing TLS callbacks: {e}")
        return {}

def analyze_dos_stub(pe) -> Dict[str, Any]:
    """Analyze DOS stub program."""
    try:
        dos_stub = {
            'exists': False,
            'size': 0,
            'entropy': 0.0,
        }

        if hasattr(pe, 'DOS_HEADER'):
            stub_offset = pe.DOS_HEADER.e_lfanew - 64  # Typical DOS stub starts after DOS header
            if stub_offset > 0:
                dos_stub_data = pe.__data__[64:pe.DOS_HEADER.e_lfanew]
                if dos_stub_data:
                    dos_stub['exists'] = True
                    dos_stub['size'] = len(dos_stub_data)
                    dos_stub['entropy'] = calculate_entropy(list(dos_stub_data))

        return dos_stub
    except Exception as ex:
          logging.error(f"Error analyzing DOS stub: {ex}")
          return {}

def analyze_certificates(pe) -> Dict[str, Any]:
    """Analyze security certificates."""
    try:
        cert_info = {}
        if hasattr(pe, 'DIRECTORY_ENTRY_SECURITY'):
            cert_info['virtual_address'] = pe.DIRECTORY_ENTRY_SECURITY.VirtualAddress
            cert_info['size'] = pe.DIRECTORY_ENTRY_SECURITY.Size

            # Extract certificate attributes if available
            if hasattr(pe, 'VS_FIXEDFILEINFO'):
                cert_info['fixed_file_info'] = {
                    'signature': pe.VS_FIXEDFILEINFO.Signature,
                    'struct_version': pe.VS_FIXEDFILEINFO.StrucVersion,
                    'file_version': f"{pe.VS_FIXEDFILEINFO.FileVersionMS >> 16}.{pe.VS_FIXEDFILEINFO.FileVersionMS & 0xFFFF}.{pe.VS_FIXEDFILEINFO.FileVersionLS >> 16}.{pe.VS_FIXEDFILEINFO.FileVersionLS & 0xFFFF}",
                    'product_version': f"{pe.VS_FIXEDFILEINFO.ProductVersionMS >> 16}.{pe.VS_FIXEDFILEINFO.ProductVersionMS & 0xFFFF}.{pe.VS_FIXEDFILEINFO.ProductVersionLS >> 16}.{pe.VS_FIXEDFILEINFO.ProductVersionLS & 0xFFFF}",
                    'file_flags': pe.VS_FIXEDFILEINFO.FileFlags,
                    'file_os': pe.VS_FIXEDFILEINFO.FileOS,
                    'file_type': pe.VS_FIXEDFILEINFO.FileType,
                    'file_subtype': pe.VS_FIXEDFILEINFO.FileSubtype,
                }

        return cert_info
    except Exception as e:
        logging.error(f"Error analyzing certificates: {e}")
        return {}

def analyze_delay_imports(pe) -> List[Dict[str, Any]]:
    """Analyze delay-load imports with error handling for missing attributes."""
    try:
        delay_imports = []
        if hasattr(pe, 'DIRECTORY_ENTRY_DELAY_IMPORT'):
            for entry in pe.DIRECTORY_ENTRY_DELAY_IMPORT:
                imports = []
                for imp in entry.imports:
                    import_info = {
                        'name': imp.name.decode() if imp.name else None,
                        'address': imp.address,
                        'ordinal': imp.ordinal,
                    }
                    imports.append(import_info)

                delay_import = {
                    'dll': entry.dll.decode() if entry.dll else None,
                    'attributes': getattr(entry.struct, 'Attributes', None),
                    'name': getattr(entry.struct, 'Name', None),
                    'handle': getattr(entry.struct, 'Handle', None),
                    'iat': getattr(entry.struct, 'IAT', None),
                    'bound_iat': getattr(entry.struct, 'BoundIAT', None),
                    'unload_iat': getattr(entry.struct, 'UnloadIAT', None),
                    'timestamp': getattr(entry.struct, 'TimeDateStamp', None),
                    'imports': imports
                }
                delay_imports.append(delay_import)

        return delay_imports
    except Exception as e:
        logging.error(f"Error analyzing delay imports: {e}")
        return []

def analyze_load_config(pe) -> Dict[str, Any]:
    """Analyze load configuration."""
    try:
        load_config = {}
        if hasattr(pe, 'DIRECTORY_ENTRY_LOAD_CONFIG'):
            config = pe.DIRECTORY_ENTRY_LOAD_CONFIG.struct
            load_config = {
                'size': config.Size,
                'timestamp': config.TimeDateStamp,
                'major_version': config.MajorVersion,
                'minor_version': config.MinorVersion,
                'global_flags_clear': config.GlobalFlagsClear,
                'global_flags_set': config.GlobalFlagsSet,
                'critical_section_default_timeout': config.CriticalSectionDefaultTimeout,
                'decommit_free_block_threshold': config.DeCommitFreeBlockThreshold,
                'decommit_total_free_threshold': config.DeCommitTotalFreeThreshold,
                'security_cookie': config.SecurityCookie,
                'se_handler_table': config.SEHandlerTable,
                'se_handler_count': config.SEHandlerCount
            }

        return load_config
    except Exception as e:
        logging.error(f"Error analyzing load config: {e}")
        return {}

def analyze_relocations(pe) -> List[Dict[str, Any]]:
    """Analyze base relocations with summarized entries."""
    try:
        relocations = []
        if hasattr(pe, 'DIRECTORY_ENTRY_BASERELOC'):
            for base_reloc in pe.DIRECTORY_ENTRY_BASERELOC:
                entry_types = {}
                offsets = []

                for entry in base_reloc.entries:
                    entry_types[entry.type] = entry_types.get(entry.type, 0) + 1
                    offsets.append(entry.rva - base_reloc.struct.VirtualAddress)

                reloc_info = {
                    'virtual_address': base_reloc.struct.VirtualAddress,
                    'size_of_block': base_reloc.struct.SizeOfBlock,
                    'summary': {
                        'total_entries': len(base_reloc.entries),
                        'types': entry_types,
                        'offset_range': (min(offsets), max(offsets)) if offsets else None
                    }
                }

                relocations.append(reloc_info)

        return relocations
    except Exception as e:
        logging.error(f"Error analyzing relocations: {e}")
        return []

def analyze_overlay(pe, file_path: str) -> Dict[str, Any]:
    """Analyze file overlay (data appended after the PE structure)."""
    try:
        overlay_info = {
            'exists': False,
            'offset': 0,
            'size': 0,
            'entropy': 0.0
        }

        last_section = max(pe.sections, key=lambda s: s.PointerToRawData + s.SizeOfRawData)
        end_of_pe = last_section.PointerToRawData + last_section.SizeOfRawData
        file_size = os.path.getsize(file_path)

        if file_size > end_of_pe:
            with open(file_path, 'rb') as f:
                f.seek(end_of_pe)
                overlay_data = f.read()

                overlay_info['exists'] = True
                overlay_info['offset'] = end_of_pe
                overlay_info['size'] = len(overlay_data)
                overlay_info['entropy'] = calculate_entropy(list(overlay_data))

        return overlay_info
    except Exception as e:
        logging.error(f"Error analyzing overlay: {e}")
        return {}

def analyze_bound_imports(pe) -> List[Dict[str, Any]]:
    """Analyze bound imports with robust error handling."""
    try:
        bound_imports = []
        if hasattr(pe, 'DIRECTORY_ENTRY_BOUND_IMPORT'):
            for bound_imp in pe.DIRECTORY_ENTRY_BOUND_IMPORT:
                bound_import = {
                    'name': bound_imp.name.decode() if bound_imp.name else None,
                    'timestamp': bound_imp.struct.TimeDateStamp,
                    'references': []
                }

                # Check if `references` exists
                if hasattr(bound_imp, 'references') and bound_imp.references:
                    for ref in bound_imp.references:
                        reference = {
                            'name': ref.name.decode() if ref.name else None,
                            'timestamp': getattr(ref.struct, 'TimeDateStamp', None)
                        }
                        bound_import['references'].append(reference)
                else:
                    logging.info(f"Bound import {bound_import['name']} has no references.")

                bound_imports.append(bound_import)

        return bound_imports
    except Exception as e:
        logging.error(f"Error analyzing bound imports: {e}")
        return []

def analyze_section_characteristics(pe) -> Dict[str, Dict[str, Any]]:
    """Analyze detailed section characteristics."""
    try:
        characteristics = {}
        for section in pe.sections:
            section_name = section.Name.decode(errors='ignore').strip('\x00')
            flags = section.Characteristics

            # Decode section characteristics flags
            section_flags = {
                'CODE': bool(flags & 0x20),
                'INITIALIZED_DATA': bool(flags & 0x40),
                'UNINITIALIZED_DATA': bool(flags & 0x80),
                'MEM_DISCARDABLE': bool(flags & 0x2000000),
                'MEM_NOT_CACHED': bool(flags & 0x4000000),
                'MEM_NOT_PAGED': bool(flags & 0x8000000),
                'MEM_SHARED': bool(flags & 0x10000000),
                'MEM_EXECUTE': bool(flags & 0x20000000),
                'MEM_READ': bool(flags & 0x40000000),
                'MEM_WRITE': bool(flags & 0x80000000)
            }

            characteristics[section_name] = {
                'flags': section_flags,
                'entropy': calculate_entropy(list(section.get_data())),
                'size_ratio': section.SizeOfRawData / pe.OPTIONAL_HEADER.SizeOfImage if pe.OPTIONAL_HEADER.SizeOfImage else 0,
                'pointer_to_raw_data': section.PointerToRawData,
                'pointer_to_relocations': section.PointerToRelocations,
                'pointer_to_line_numbers': section.PointerToLinenumbers,
                'number_of_relocations': section.NumberOfRelocations,
                'number_of_line_numbers': section.NumberOfLinenumbers,
            }

        return characteristics
    except Exception as e:
        logging.error(f"Error analyzing section characteristics: {e}")
        return {}

def analyze_extended_headers(pe) -> Dict[str, Any]:
    """Analyze extended header information."""
    try:
        headers = {
            'dos_header': {
                'e_magic': pe.DOS_HEADER.e_magic,
                'e_cblp': pe.DOS_HEADER.e_cblp,
                'e_cp': pe.DOS_HEADER.e_cp,
                'e_crlc': pe.DOS_HEADER.e_crlc,
                'e_cparhdr': pe.DOS_HEADER.e_cparhdr,
                'e_minalloc': pe.DOS_HEADER.e_minalloc,
                'e_maxalloc': pe.DOS_HEADER.e_maxalloc,
                'e_ss': pe.DOS_HEADER.e_ss,
                'e_sp': pe.DOS_HEADER.e_sp,
                'e_csum': pe.DOS_HEADER.e_csum,
                'e_ip': pe.DOS_HEADER.e_ip,
                'e_cs': pe.DOS_HEADER.e_cs,
                'e_lfarlc': pe.DOS_HEADER.e_lfarlc,
                'e_ovno': pe.DOS_HEADER.e_ovno,
                'e_oemid': pe.DOS_HEADER.e_oemid,
                'e_oeminfo': pe.DOS_HEADER.e_oeminfo
            },
            'nt_headers': {}
        }

        # Ensure NT_HEADERS exists and contains FileHeader
        if hasattr(pe, 'NT_HEADERS') and pe.NT_HEADERS is not None:
            nt_headers = pe.NT_HEADERS
            if hasattr(nt_headers, 'FileHeader'):
                headers['nt_headers'] = {
                    'signature': nt_headers.Signature,
                    'machine': nt_headers.FileHeader.Machine,
                    'number_of_sections': nt_headers.FileHeader.NumberOfSections,
                    'time_date_stamp': nt_headers.FileHeader.TimeDateStamp,
                    'characteristics': nt_headers.FileHeader.Characteristics
                }

        return headers
    except Exception as e:
        logging.error(f"Error analyzing extended headers: {e}")
        return {}

def serialize_data(data) -> Any:
    """Serialize data for output, ensuring compatibility."""
    try:
        return list(data) if data else None
    except Exception:
        return None

def analyze_rich_header(pe) -> Dict[str, Any]:
    """Analyze Rich header details."""
    try:
        rich_header = {}
        if hasattr(pe, 'RICH_HEADER') and pe.RICH_HEADER is not None:
            rich_header['checksum'] = getattr(pe.RICH_HEADER, 'checksum', None)
            rich_header['values'] = serialize_data(pe.RICH_HEADER.values)
            rich_header['clear_data'] = serialize_data(pe.RICH_HEADER.clear_data)
            rich_header['key'] = serialize_data(pe.RICH_HEADER.key)
            rich_header['raw_data'] = serialize_data(pe.RICH_HEADER.raw_data)

            # Decode CompID and build number information
            compid_info = []
            for i in range(0, len(pe.RICH_HEADER.values), 2):
                if i + 1 < len(pe.RICH_HEADER.values):
                    comp_id = pe.RICH_HEADER.values[i] >> 16
                    build_number = pe.RICH_HEADER.values[i] & 0xFFFF
                    count = pe.RICH_HEADER.values[i + 1]
                    compid_info.append({
                        'comp_id': comp_id,
                        'build_number': build_number,
                        'count': count
                    })
            rich_header['comp_id_info'] = compid_info

        return rich_header
    except Exception as e:
        logging.error(f"Error analyzing Rich header: {e}")
        return {}

def extract_numeric_features(file_path: str, rank: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Extract numeric features of a file using pefile.
    """
    try:
        # Load the PE file
        pe = pefile.PE(file_path)

        # Extract features
        numeric_features = {
            # Optional Header Features
            'SizeOfOptionalHeader': pe.FILE_HEADER.SizeOfOptionalHeader,
            'MajorLinkerVersion': pe.OPTIONAL_HEADER.MajorLinkerVersion,
            'MinorLinkerVersion': pe.OPTIONAL_HEADER.MinorLinkerVersion,
            'SizeOfCode': pe.OPTIONAL_HEADER.SizeOfCode,
            'SizeOfInitializedData': pe.OPTIONAL_HEADER.SizeOfInitializedData,
            'SizeOfUninitializedData': pe.OPTIONAL_HEADER.SizeOfUninitializedData,
            'AddressOfEntryPoint': pe.OPTIONAL_HEADER.AddressOfEntryPoint,
            'BaseOfCode': pe.OPTIONAL_HEADER.BaseOfCode,
            'BaseOfData': getattr(pe.OPTIONAL_HEADER, 'BaseOfData', 0),
            'ImageBase': pe.OPTIONAL_HEADER.ImageBase,
            'SectionAlignment': pe.OPTIONAL_HEADER.SectionAlignment,
            'FileAlignment': pe.OPTIONAL_HEADER.FileAlignment,
            'MajorOperatingSystemVersion': pe.OPTIONAL_HEADER.MajorOperatingSystemVersion,
            'MinorOperatingSystemVersion': pe.OPTIONAL_HEADER.MinorOperatingSystemVersion,
            'MajorImageVersion': pe.OPTIONAL_HEADER.MajorImageVersion,
            'MinorImageVersion': pe.OPTIONAL_HEADER.MinorImageVersion,
            'MajorSubsystemVersion': pe.OPTIONAL_HEADER.MajorSubsystemVersion,
            'MinorSubsystemVersion': pe.OPTIONAL_HEADER.MinorSubsystemVersion,
            'SizeOfImage': pe.OPTIONAL_HEADER.SizeOfImage,
            'SizeOfHeaders': pe.OPTIONAL_HEADER.SizeOfHeaders,
            'CheckSum': pe.OPTIONAL_HEADER.CheckSum,
            'Subsystem': pe.OPTIONAL_HEADER.Subsystem,
            'DllCharacteristics': pe.OPTIONAL_HEADER.DllCharacteristics,
            'SizeOfStackReserve': pe.OPTIONAL_HEADER.SizeOfStackReserve,
            'SizeOfStackCommit': pe.OPTIONAL_HEADER.SizeOfStackCommit,
            'SizeOfHeapReserve': pe.OPTIONAL_HEADER.SizeOfHeapReserve,
            'SizeOfHeapCommit': pe.OPTIONAL_HEADER.SizeOfHeapCommit,
            'LoaderFlags': pe.OPTIONAL_HEADER.LoaderFlags,
            'NumberOfRvaAndSizes': pe.OPTIONAL_HEADER.NumberOfRvaAndSizes,

            # Section Headers
            'sections': [
                {
                    'name': section.Name.decode(errors='ignore').strip('\x00'),
                    'virtual_size': section.Misc_VirtualSize,
                    'virtual_address': section.VirtualAddress,
                    'size_of_raw_data': section.SizeOfRawData,
                    'pointer_to_raw_data': section.PointerToRawData,
                    'characteristics': section.Characteristics,
                }
                for section in pe.sections
            ],

            # Imported Functions
            'imports': [
                imp.name.decode(errors='ignore') if imp.name else "Unknown"
                for entry in getattr(pe, 'DIRECTORY_ENTRY_IMPORT', [])
                for imp in getattr(entry, 'imports', [])
            ] if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT') else [],

            # Exported Functions
            'exports': [
                exp.name.decode(errors='ignore') if exp.name else "Unknown"
                for exp in getattr(getattr(pe, 'DIRECTORY_ENTRY_EXPORT', None), 'symbols', [])
            ] if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT') else [],

            # Resources
            'resources': [
                {
                    'type_id': getattr(getattr(resource_type, 'struct', None), 'Id', None),
                    'resource_id': getattr(getattr(resource_id, 'struct', None), 'Id', None),
                    'lang_id': getattr(getattr(resource_lang, 'struct', None), 'Id', None),
                    'size': getattr(getattr(resource_lang, 'data', None), 'Size', None),
                    'codepage': getattr(getattr(resource_lang, 'data', None), 'CodePage', None),
                }
                for resource_type in
                (pe.DIRECTORY_ENTRY_RESOURCE.entries if hasattr(pe.DIRECTORY_ENTRY_RESOURCE, 'entries') else [])
                for resource_id in (resource_type.directory.entries if hasattr(resource_type, 'directory') else [])
                for resource_lang in (resource_id.directory.entries if hasattr(resource_id, 'directory') else [])
                if hasattr(resource_lang, 'data')
            ] if hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE') else [],

            # Certificates
            'certificates': analyze_certificates(pe),  # Analyze certificates

            # DOS Stub Analysis
            'dos_stub': analyze_dos_stub(pe),  # DOS stub analysis here

            # TLS Callbacks
            'tls_callbacks': analyze_tls_callbacks(pe),  # TLS callback analysis here

            # Delay Imports
            'delay_imports': analyze_delay_imports(pe),  # Delay imports analysis here

            # Load Config
            'load_config': analyze_load_config(pe),  # Load config analysis here

            # Relocations
            'relocations': analyze_relocations(pe),  # Relocations analysis here

            # Bound Imports
            'bound_imports': analyze_bound_imports(pe),  # Bound imports analysis here

            # Section Characteristics
            'section_characteristics': analyze_section_characteristics(pe),  # Section characteristics analysis here

            # Extended Headers
            'extended_headers': analyze_extended_headers(pe),  # Extended headers analysis here

            # Rich Header
            'rich_header': analyze_rich_header(pe),  # Rich header analysis here

            # Overlay
            'overlay': analyze_overlay(pe, file_path),  # Overlay analysis here
        }

        # Add numeric tag if provided
        if rank is not None:
            numeric_features['numeric_tag'] = rank

        return numeric_features

    except Exception as ex:
        logging.error(f"Error extracting numeric features from {file_path}: {str(ex)}", exc_info=True)
        return None

def calculate_similarity(features1, features2):
    """Calculate similarity between two dictionaries of features"""
    common_keys = set(features1.keys()) & set(features2.keys())
    matching_keys = sum(1 for key in common_keys if features1[key] == features2[key])
    similarity = matching_keys / max(len(features1), len(features2))
    return similarity

# a global (or outer-scope) list to collect every saved path
saved_paths = []
saved_pyc_paths = []
deobfuscated_saved_paths = []
path_lists = [saved_paths, deobfuscated_saved_paths, saved_pyc_paths]

def scan_file_with_machine_learning_ai(file_path, threshold=0.86):
    """Scan a file for malicious activity using machine learning definitions loaded from JSON."""

    # Default assignment of malware_definition before starting the process
    malware_definition = "Unknown"  # Assume unknown until checked
    logging.info(f"Starting machine learning scan for file: {file_path}")

    try:
        pe = pefile.PE(file_path)
        if not pe:
            logging.warning(f"File {file_path} is not a valid PE file. Returning default value 'Unknown'.")
            return False, malware_definition, 0

        logging.info(f"File {file_path} is a valid PE file, proceeding with feature extraction.")
        file_info = extract_infos(file_path)
        file_numeric_features = extract_numeric_features(file_path)

        is_malicious_ml = False
        nearest_malicious_similarity = 0
        nearest_benign_similarity = 0

        logging.info(f"File information: {file_info}")

        # Check malicious definitions
        for ml_feats, info in zip(malicious_numeric_features, malicious_file_names):
            rank = info['numeric_tag']
            similarity = calculate_similarity(file_numeric_features, ml_feats)
            nearest_malicious_similarity = max(nearest_malicious_similarity, similarity)
            if similarity >= threshold:
                is_malicious_ml = True
                malware_definition = info['file_name']
                logging.warning(f"Malicious activity detected in {file_path}. Definition: {malware_definition}, similarity: {similarity}, rank: {rank}")

        # If not malicious, check benign
        if not is_malicious_ml:
            for ml_feats, info in zip(benign_numeric_features, benign_file_names):
                similarity = calculate_similarity(file_numeric_features, ml_feats)
                nearest_benign_similarity = max(nearest_benign_similarity, similarity)
                benign_definition = info['file_name']

            if nearest_benign_similarity >= 0.93:
                malware_definition = "Benign"
                logging.info(f"File {file_path} is classified as benign ({benign_definition}) with similarity: {nearest_benign_similarity}")
            else:
                malware_definition = "Unknown"
                logging.info(f"File {file_path} is classified as unknown with similarity: {nearest_benign_similarity}")

        # Return result
        if is_malicious_ml:
            return False, malware_definition, nearest_malicious_similarity
        else:
            return False, malware_definition, nearest_benign_similarity

    except pefile.PEFormatError:
        logging.error(f"Error: {file_path} does not have a valid PE format.")
        return False, malware_definition, 0
    except Exception as ex:
        logging.error(f"An error occurred while scanning file {file_path}: {ex}")
        return False, malware_definition, 0

def restart_clamd_thread():
    try:
        threading.Thread(target=restart_clamd).start()
    except Exception as ex:
        logging.error(f"Error starting clamd restart thread: {ex}")

def restart_clamd():
    try:
        logging.info("Stopping ClamAV...")
        stop_result = subprocess.run(["net", "stop", 'clamd'], capture_output=True, text=True, encoding="utf-8", errors="ignore")
        if stop_result.returncode != 0:
                logging.error("Failed to stop ClamAV.")

        logging.info("Starting ClamAV...")
        start_result = subprocess.run(["net", "start", 'clamd'], capture_output=True, text=True, encoding="utf-8", errors="ignore")
        if start_result.returncode == 0:
            logging.info("ClamAV restarted successfully.")
            return True
        else:
            logging.error("Failed to start ClamAV.")
            return False
    except Exception as ex:
        logging.error(f"An error occurred while restarting ClamAV: {ex}")
        return False

def scan_file_with_clamd(file_path):
    """Scan file using clamd."""
    try:
        file_path = os.path.abspath(file_path)  # Get absolute path
        result = subprocess.run([clamdscan_path, file_path], capture_output=True, text=True, encoding="utf-8", errors="ignore")
        clamd_output = result.stdout
        logging.info(f"Clamdscan output: {clamd_output}")

        if "ERROR" in clamd_output:
            logging.info(f"Clamdscan reported an error: {clamd_output}")
            return "Clean"
        elif "FOUND" in clamd_output:
            match = re.search(r": (.+) FOUND", clamd_output)
            if match:
                virus_name = match.group(1).strip()
                return virus_name
        elif "OK" in clamd_output or "Infected files: 0" in clamd_output:
            return "Clean"
        else:
            logging.info(f"Unexpected clamdscan output: {clamd_output}")
            return "Clean"
    except Exception as ex:
        logging.error(f"Error scanning file {file_path}: {ex}")
        return "Clean"

def scan_yara(file_path):
    matched_rules = []

    try:
        if not os.path.exists(file_path):
            logging.error(f"File not found during YARA scan: {file_path}")
            return None

        with open(file_path, 'rb') as yara_file:
            data_content = yara_file.read()

            # compiled_rule
            try:
                if compiled_rule:
                    matches = compiled_rule.match(data=data_content)
                    for match in matches or []:
                        if match.rule not in excluded_rules:
                            matched_rules.append(match.rule)
                        else:
                            logging.info(f"Rule {match.rule} is excluded from compiled_rule.")
                else:
                    logging.error("compiled_rule is not defined.")
            except Exception as e:
                logging.error(f"Error scanning with compiled_rule: {e}")

            # yarGen_rule
            try:
                if yarGen_rule:
                    matches = yarGen_rule.match(data=data_content)
                    for match in matches or []:
                        if match.rule not in excluded_rules:
                            matched_rules.append(match.rule)
                        else:
                            logging.info(f"Rule {match.rule} is excluded from yarGen_rule.")
                else:
                    logging.error("yarGen_rule is not defined.")
            except Exception as e:
                logging.error(f"Error scanning with yarGen_rule: {e}")

            # icewater_rule
            try:
                if icewater_rule:
                    matches = icewater_rule.match(data=data_content)
                    for match in matches or []:
                        if match.rule not in excluded_rules:
                            matched_rules.append(match.rule)
                        else:
                            logging.info(f"Rule {match.rule} is excluded from icewater_rule.")
                else:
                    logging.error("icewater_rule is not defined.")
            except Exception as e:
                logging.error(f"Error scanning with icewater_rule: {e}")

            # valhalla_rule
            try:
                if valhalla_rule:
                    matches = valhalla_rule.match(data=data_content)
                    for match in matches or []:
                        if match.rule not in excluded_rules:
                            matched_rules.append(match.rule)
                        else:
                            logging.info(f"Rule {match.rule} is excluded from valhalla_rule.")
                else:
                    logging.error("valhalla_rule is not defined.")
            except Exception as e:
                logging.error(f"Error scanning with valhalla_rule: {e}")

            # yaraxtr_rule (YARA-X)
            try:
                if yaraxtr_rule:
                    scanner = yara_x.Scanner(rules=yaraxtr_rule)
                    results = scanner.scan(data=data_content)
                    for rule in getattr(results, 'matching_rules', []) or []:
                        identifier = getattr(rule, 'identifier', None)
                        if identifier and identifier not in excluded_rules:
                            matched_rules.append(identifier)
                        else:
                            logging.info(f"Rule {identifier} is excluded from yaraxtr_rule.")
                else:
                    logging.error("yaraxtr_rule is not defined.")
            except Exception as e:
                logging.error(f"Error scanning with yaraxtr_rule: {e}")

        return matched_rules if matched_rules else None

    except Exception as ex:
        logging.error(f"An error occurred during YARA scan: {ex}")
        return None

def detect_etw_tampering_sandbox(moved_sandboxed_ntdll_path):
    """
    Compare the NtTraceEvent bytes in the sandboxed ntdll.dll file against the original
    on-disk ntdll.dll in System32.
    Logs a warning if the sandboxed copy is tampered (bytes differ).
    Returns True if tampered, False otherwise.
    """
    try:
        if not os.path.isfile(ntdll_path):
            logging.error(f"[ETW Sandbox Detection] Original ntdll.dll not found at {ntdll_path}")
            return False
        if not os.path.isfile(moved_sandboxed_ntdll_path):
            logging.error(f"[ETW Sandbox Detection] Sandboxed ntdll.dll not found at {moved_sandboxed_ntdll_path}")
            return False

        # Load original PE to find NtTraceEvent RVA
        try:
            pe_orig = pefile.PE(ntdll_path, fast_load=True)
            pe_orig.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXPORT']])
        except Exception as e:
            logging.error(f"[ETW Sandbox Detection] Failed to parse original PE: {e}")
            return False

        nttrace_rva = None
        for exp in getattr(pe_orig, 'DIRECTORY_ENTRY_EXPORT', []).symbols:
            if exp.name and exp.name.decode(errors='ignore') == "NtTraceEvent":
                nttrace_rva = exp.address
                break
        if nttrace_rva is None:
            logging.error("[ETW Sandbox Detection] Export NtTraceEvent not found in original ntdll.dll")
            return False

        # Compute offset in original file
        try:
            orig_offset = pe_orig.get_offset_from_rva(nttrace_rva)
        except Exception as e:
            logging.error(f"[ETW Sandbox Detection] Cannot compute offset in original for RVA {hex(nttrace_rva)}: {e}")
            return False

        # Load sandboxed PE to compute offset there
        try:
            pe_sandbox = pefile.PE(moved_sandboxed_ntdll_path, fast_load=True)
            pe_sandbox.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXPORT']])
        except Exception as e:
            logging.error(f"[ETW Sandbox Detection] Failed to parse sandboxed PE: {e}")
            return False

        # Verify that sandboxed export table contains NtTraceEvent (optional but good)
        found_in_sandbox = False
        for exp in getattr(pe_sandbox, 'DIRECTORY_ENTRY_EXPORT', []).symbols:
            if exp.name and exp.name.decode(errors='ignore') == "NtTraceEvent":
                found_in_sandbox = True
                break
        if not found_in_sandbox:
            logging.error("[ETW Sandbox Detection] Export NtTraceEvent not found in sandboxed ntdll.dll")
            return False

        # Compute offset in sandboxed file
        try:
            sandbox_offset = pe_sandbox.get_offset_from_rva(nttrace_rva)
        except Exception as e:
            logging.error(f"[ETW Sandbox Detection] Cannot compute offset in sandboxed for RVA {hex(nttrace_rva)}: {e}")
            return False

        # Read bytes
        length = 16
        try:
            with open(ntdll_path, "rb") as f_orig:
                f_orig.seek(orig_offset)
                orig_bytes = f_orig.read(length)
            if len(orig_bytes) < length:
                logging.error(f"[ETW Sandbox Detection] Could not read {length} bytes from original ntdll.dll")
                return False
        except Exception as e:
            logging.error(f"[ETW Sandbox Detection] Error reading original file: {e}")
            return False

        try:
            with open(moved_sandboxed_ntdll_path, "rb") as f_s:
                f_s.seek(sandbox_offset)
                sandbox_bytes = f_s.read(length)
            if len(sandbox_bytes) < length:
                logging.error(f"[ETW Sandbox Detection] Could not read {length} bytes from sandboxed ntdll.dll")
                return False
        except Exception as e:
            logging.error(f"[ETW Sandbox Detection] Error reading sandboxed file: {e}")
            return False

        # Compare
        if sandbox_bytes != orig_bytes:
            orig_hex = orig_bytes[:8].hex()
            sand_hex = sandbox_bytes[:8].hex()
            logging.warning(
                f"[ETW Sandbox Detection] Sandboxed ntdll.dll NtTraceEvent seems patched: "
                f"original bytes={orig_hex}, sandbox bytes={sand_hex}"
            )
            return True

        # No tampering detected
        return False

    except Exception as ex:
        logging.error(f"[ETW Sandbox Detection] Unexpected error: {ex}")
        return False

# Constants for CryptQueryObject
CERT_QUERY_OBJECT_FILE = 0x00000001
CERT_QUERY_CONTENT_FLAG_PKCS7_SIGNED_EMBED = 0x00000080
CERT_QUERY_FORMAT_FLAG_BINARY = 0x00000002

# CertGetNameStringW flags/types
CERT_NAME_SIMPLE_DISPLAY_TYPE = 4
CERT_NAME_ISSUER_FLAG = 1

crypt32 = ctypes.windll.crypt32

# Define CERT_CONTEXT struct for extracting raw encoded certificate bytes
class CERT_CONTEXT(ctypes.Structure):
    _fields_ = [
        ("dwCertEncodingType", wintypes.DWORD),
        ("pbCertEncoded", ctypes.POINTER(ctypes.c_byte)),
        ("cbCertEncoded", wintypes.DWORD),
        ("pCertInfo", ctypes.c_void_p),
        ("hCertStore", ctypes.c_void_p),
    ]

PCCERT_CONTEXT = ctypes.POINTER(CERT_CONTEXT)

# HRESULT codes for "no signature" cases
TRUST_E_NOSIGNATURE = 0x800B0100
TRUST_E_SUBJECT_FORM_UNKNOWN = 0x800B0008
TRUST_E_PROVIDER_UNKNOWN     = 0x800B0001
NO_SIGNATURE_CODES = {
    TRUST_E_NOSIGNATURE,
    TRUST_E_SUBJECT_FORM_UNKNOWN,
    TRUST_E_PROVIDER_UNKNOWN,
}

# Constants for WinVerifyTrust
class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]

WINTRUST_ACTION_GENERIC_VERIFY_V2 = GUID(
    0x00AAC56B, 0xCD44, 0x11D0,
    (ctypes.c_ubyte * 8)(0x8C, 0xC2, 0x00, 0xC0, 0x4F, 0xC2, 0x95, 0xEE)
)

class WINTRUST_FILE_INFO(ctypes.Structure):
    _fields_ = [
        ("cbStruct", wintypes.DWORD),
        ("pcwszFilePath", wintypes.LPCWSTR),
        ("hFile", wintypes.HANDLE),
        ("pgKnownSubject", ctypes.POINTER(GUID)),
    ]

class WINTRUST_DATA(ctypes.Structure):
    _fields_ = [
        ("cbStruct", wintypes.DWORD),
        ("pPolicyCallbackData", ctypes.c_void_p),
        ("pSIPClientData", ctypes.c_void_p),
        ("dwUIChoice", wintypes.DWORD),
        ("fdwRevocationChecks", wintypes.DWORD),
        ("dwUnionChoice", wintypes.DWORD),
        ("pFile", ctypes.POINTER(WINTRUST_FILE_INFO)),
        ("dwStateAction", wintypes.DWORD),
        ("hWVTStateData", wintypes.HANDLE),
        ("pwszURLReference", wintypes.LPCWSTR),
        ("dwProvFlags", wintypes.DWORD),
        ("dwUIContext", wintypes.DWORD),
        ("pSignatureSettings", ctypes.c_void_p),
    ]

# UI and revocation options
WTD_UI_NONE = 2
WTD_REVOKE_NONE = 0
WTD_CHOICE_FILE = 1
WTD_STATEACTION_IGNORE = 0x00000000

# Load WinTrust DLL
_wintrust = ctypes.windll.wintrust


def _build_wtd_for(file_path: str) -> WINTRUST_DATA:
    """Internal helper to populate a WINTRUST_DATA for the given file."""
    file_info = WINTRUST_FILE_INFO(
        ctypes.sizeof(WINTRUST_FILE_INFO), file_path, None, None
    )
    wtd = WINTRUST_DATA()
    ctypes.memset(ctypes.byref(wtd), 0, ctypes.sizeof(wtd))
    wtd.cbStruct = ctypes.sizeof(WINTRUST_DATA)
    wtd.dwUIChoice = WTD_UI_NONE
    wtd.fdwRevocationChecks = WTD_REVOKE_NONE
    wtd.dwUnionChoice = WTD_CHOICE_FILE
    wtd.pFile = ctypes.pointer(file_info)
    wtd.dwStateAction = WTD_STATEACTION_IGNORE
    return wtd


def verify_authenticode_signature(file_path: str) -> int:
    """Calls WinVerifyTrust and returns the raw HRESULT."""
    wtd = _build_wtd_for(file_path)
    return _wintrust.WinVerifyTrust(
        None,
        ctypes.byref(WINTRUST_ACTION_GENERIC_VERIFY_V2),
        ctypes.byref(wtd)
    )

def check_valid_signature(file_path: str) -> dict:
    """
    Returns {"is_valid": bool, "status": str}.
    """
    try:
        result = verify_authenticode_signature(file_path)

        if result == 0:
            is_valid = True
            status = "Valid"
        elif result in NO_SIGNATURE_CODES:
            is_valid = False
            status = "No signature"
        else:
            is_valid = False
            status = "Invalid signature"

        return {"is_valid": is_valid, "status": status}
    except Exception as ex:
        logging.error(f"[Signature] {file_path}: {ex}")
        return {"is_valid": False, "status": str(ex)}

def is_encrypted(zip_info):
    """Check if a ZIP entry is encrypted."""
    return zip_info.flag_bits & 0x1 != 0

def contains_rlo_after_dot_with_extension_check(filename, fileTypes):
    """
    Check if the filename contains an RLO character after a dot AND has a known extension.
    This helps detect potential RLO attacks that try to disguise malicious files.

    Args:
        filename (str): The filename to check
        fileTypes (set/list): Collection of known/allowed file extensions

    Returns:
        bool: True if RLO found after dot AND file has known extension, False otherwise
    """
    try:
        # First check if there's an RLO character after a dot
        if ".\u202E" not in filename:
            return False
        # If RLO found after dot, check if file has a known extension
        ext = os.path.splitext(filename)[1].lower()
        logging.info(f"RLO detected after dot in '{filename}', checking extension '{ext}'")
        has_known_ext = ext in fileTypes
        if has_known_ext:
            logging.warning(f"POTENTIAL RLO ATTACK: File '{filename}' has RLO after dot with known extension '{ext}'")
        else:
            logging.info(f"RLO found after dot but extension '{ext}' not in known types")
        return has_known_ext
    except Exception as ex:
        logging.error(f"Error checking RLO and extension for file {filename}: {ex}")
        return False

def detect_suspicious_filename_patterns(filename, fileTypes, max_spaces=10):
    """
    Detect various filename obfuscation techniques including:
    - RLO (Right-to-Left Override) attacks
    - Excessive spaces to hide real extensions
    - Multiple extensions

    Args:
        filename (str): The filename to check
        fileTypes (set/list): Collection of known/allowed file extensions
        max_spaces (int): Maximum allowed consecutive spaces

    Returns:
        dict: Detection results with attack types found
    """
    results = {
        'rlo_attack': False,
        'excessive_spaces': False,
        'multiple_extensions': False,
        'suspicious': False,
        'details': []
    }

    try:
        # Check for RLO attack
        if ".\u202E" in filename:
            ext = os.path.splitext(filename)[1].lower()
            if ext in fileTypes:
                results['rlo_attack'] = True
                results['details'].append(f"RLO character found after dot with known extension '{ext}'")

        # Check for excessive spaces (potential extension hiding)
        if '  ' in filename:  # Start with double space check
            space_count = 0
            max_consecutive_spaces = 0

            for char in filename:
                if char == ' ':
                    space_count += 1
                    max_consecutive_spaces = max(max_consecutive_spaces, space_count)
                else:
                    space_count = 0

            if max_consecutive_spaces > max_spaces:
                results['excessive_spaces'] = True
                results['details'].append(f"Excessive spaces detected: {max_consecutive_spaces} consecutive spaces")

                # Check if there's a hidden extension after the spaces
                trimmed_filename = filename.rstrip()
                if trimmed_filename != filename:
                    hidden_ext = os.path.splitext(trimmed_filename)[1].lower()
                    if hidden_ext in fileTypes:
                        results['details'].append(f"Potential hidden extension: '{hidden_ext}'")

        # Check for multiple extensions (only flag if more than 4 extensions)
        parts = filename.split('.')
        if len(parts) > 5:  # More than 4 extensions (5 parts = filename + 4 extensions)
            extensions = ['.' + part.lower() for part in parts[1:]]
            known_extensions = [ext for ext in extensions if ext in fileTypes]

            if known_extensions:  # Only flag if there are known extensions
                results['multiple_extensions'] = True
                results['details'].append(f"Excessive extensions detected ({len(parts)-1} extensions): {known_extensions}")

        # Mark as suspicious if any attack detected
        results['suspicious'] = any([
            results['rlo_attack'],
            results['excessive_spaces'],
            results['multiple_extensions']
        ])

        if results['suspicious']:
            logging.warning(f"SUSPICIOUS FILENAME DETECTED: {filename} - {results['details']}")

        return results

    except Exception as ex:
        logging.error(f"Error analyzing filename {filename}: {ex}")
        return results

def scan_zip_file(file_path):
    """
    Scan a ZIP archive for:
      - RLO in filename warnings (encrypted vs non-encrypted)
      - Size bomb warnings (even if AES encrypted)
      - Single entry text files containing"Password:" (HEUR:Win32.Susp.Encrypted.Zip.SingleEntry)

    Returns:
      (success: bool, entries: List[(filename, uncompressed_size, encrypted_flag)])
    """
    try:
        zip_size = os.path.getsize(file_path)
        entries = []

        with pyzipper.ZipFile(file_path, 'r') as zf:
            for info in zf.infolist():
                encrypted = bool(info.flag_bits & 0x1)

                detection_result = detect_suspicious_filename_patterns(info.filename, fileTypes)
                if detection_result['suspicious']:
                    # Build attack type string
                    attack_types = []
                    if detection_result['rlo_attack']:
                        attack_types.append("RLO")
                    if detection_result['excessive_spaces']:
                        attack_types.append("Spaces")
                    if detection_result['multiple_extensions']:
                        attack_types.append("MultiExt")

                    attack_string = "+".join(attack_types) if attack_types else "Generic"
                    virus = f"HEUR:{attack_string}.Susp.Name.Encrypted.ZIP.gen" if encrypted else f"HEUR:{attack_string}.Susp.Name.ZIP.gen"

                    notify_susp_archive_file_name_warning(file_path, "ZIP", virus)

                # Record metadata
                entries.append((info.filename, info.file_size, encrypted))

                # Size-bomb check
                if zip_size < 20 * 1024 * 1024 and info.file_size > 650 * 1024 * 1024:
                    virus = "HEUR:Win32.Susp.Size.Encrypted.ZIP" if encrypted else "HEUR:Win32.Susp.Size.ZIP"
                    notify_size_warning(file_path, "ZIP", virus)

        # Single-entry password logic
        if len(entries) == 1:
            fname, _, encrypted = entries[0]
            if not encrypted:
                with pyzipper.ZipFile(file_path, 'r') as zf:
                    snippet = zf.open(fname).read(4096)
                if is_plain_text(snippet) and 'Password:' in snippet.decode('utf-8', errors='ignore'):
                    notify_size_warning(file_path, "ZIP", "HEUR:Win32.Susp.Encrypted.Zip.SingleEntry")

        return True, entries

    except pyzipper.zipfile.BadZipFile:
        logging.error(f"Not a valid ZIP archive: {file_path}")
        return False, []
    except Exception as ex:
        logging.error(f"Error scanning zip file: {file_path} {ex}")
        return False, []


def scan_7z_file(file_path):
    """
    Scan a 7z archive for:
      - RLO in filename warnings (encrypted vs non-encrypted)
      - Size bomb warnings (even if encrypted)
      - Single entry text files containing"Password:" (HEUR:Win32.Susp.Encrypted.7z.SingleEntry)

    Returns:
      (success: bool, entries: List[(filename, uncompressed_size, encrypted_flag)])
    """
    try:
        archive_size = os.path.getsize(file_path)
        entries = []

        with py7zr.SevenZipFile(file_path, mode='r') as archive:
            for entry in archive.list():
                filename = entry.filename
                encrypted = entry.is_encrypted

                detection_result = detect_suspicious_filename_patterns(filename, fileTypes)
                if detection_result['suspicious']:
                    # Build attack type string
                    attack_types = []
                    if detection_result['rlo_attack']:
                        attack_types.append("RLO")
                    if detection_result['excessive_spaces']:
                        attack_types.append("Spaces")
                    if detection_result['multiple_extensions']:
                        attack_types.append("MultiExt")

                    attack_string = "+".join(attack_types) if attack_types else "Generic"
                    virus = f"HEUR:{attack_string}.Susp.Name.Encrypted.7z.gen" if encrypted else f"HEUR:{attack_string}.Susp.Name.7z.gen"

                    notify_susp_archive_file_name_warning(file_path, "7z", virus)

                # Record metadata
                entries.append((filename, entry.uncompressed, encrypted))

                # Size-bomb check
                if archive_size < 20 * 1024 * 1024 and entry.uncompressed > 650 * 1024 * 1024:
                    virus = "HEUR:Win32.Susp.Size.Encrypted.7z" if encrypted else "HEUR:Win32.Susp.Size.7z"
                    notify_size_warning(file_path, "7z", virus)

        # Single-entry password logic
        if len(entries) == 1:
            fname, _, encrypted = entries[0]
            if not encrypted:
                data_map = archive.read([fname])
                snippet = data_map.get(fname, b'')[:4096]
                if is_plain_text(snippet) and 'Password:' in snippet.decode('utf-8', errors='ignore'):
                    notify_size_warning(file_path, "7z", "HEUR:Win32.Susp.Encrypted.7z.SingleEntry")

        return True, entries

    except py7zr.exceptions.Bad7zFile:
        logging.error(f"Not a valid 7z archive: {file_path}")
        return False, []
    except Exception as ex:
        logging.error(f"Error scanning 7z file: {file_path} {ex}")
        return False, []

def is_7z_file_from_output(die_output: str) -> bool:
    """
    Checks if DIE output indicates a 7-Zip archive.
    Expects the raw stdout (or equivalent) from a Detect It Easy run.
    """
    if die_output and "Archive: 7-Zip" in die_output:
        logging.info("DIE output indicates a 7z archive.")
        return True

    logging.info(f"DIE output does not indicate a 7z archive: {die_output!r}")
    return False

def scan_tar_file(file_path):
    """Scan files within a tar archive."""
    try:
        tar_size = os.path.getsize(file_path)

        with tarfile.open(file_path, 'r') as tar:
            for member in tar.getmembers():
                detection_result = detect_suspicious_filename_patterns(member.name, fileTypes)
                if detection_result['suspicious']:
                    # Build attack type string
                    attack_types = []
                    if detection_result['rlo_attack']:
                        attack_types.append("RLO")
                    if detection_result['excessive_spaces']:
                        attack_types.append("Spaces")
                    if detection_result['multiple_extensions']:
                        attack_types.append("MultiExt")

                    attack_string = "+".join(attack_types) if attack_types else "Generic"
                    virus_name = f"HEUR:{attack_string}.Susp.Name.TAR.gen"

                    logging.warning(
                        f"Filename '{member.name}' in archive '{file_path}' contains suspicious pattern(s): {attack_string} - "
                        f"flagged as {virus_name}"
                    )
                    notify_susp_archive_file_name_warning(file_path, "TAR", virus_name)

                if member.isreg():  # Check if it's a regular file
                    extracted_file_path = os.path.join(tar_extracted_dir, member.name)

                    # Skip if the file has already been processed
                    if os.path.isfile(extracted_file_path):
                        logging.info(f"File {member.name} already processed, skipping...")
                        continue

                    # Extract the file
                    tar.extract(member, tar_extracted_dir)

                    # Check for suspicious conditions: large files in small TAR archives
                    extracted_file_size = os.path.getsize(extracted_file_path)
                    if tar_size < 20 * 1024 * 1024 and extracted_file_size > 650 * 1024 * 1024:
                        virus_name = "HEUR:Win32.Susp.Size.Encrypted.TAR"
                        logging.warning(
                            f"TAR file {file_path} is smaller than 20MB but contains a large file: {member.name} "
                            f"({extracted_file_size / (1024 * 1024):.2f} MB) - flagged as {virus_name}. "
                            "Potential TARbomb or Fake Size detected to avoid VirusTotal detections."
                        )
                        notify_size_warning(file_path, "TAR", virus_name)

        return True, []
    except Exception as ex:
        logging.error(f"Error scanning tar file: {file_path} - {ex}")
        return False, ""

# Global variables for worm detection
worm_alerted_files = []
worm_detected_count = {}
worm_file_paths = []

def calculate_similarity_worm(features1, features2):
    """
    Calculate similarity between two dictionaries of features for worm detection.
    Adjusted threshold for worm detection.
    """
    try:
        common_keys = set(features1.keys()) & set(features2.keys())
        matching_keys = sum(1 for key in common_keys if features1[key] == features2[key])
        similarity = matching_keys / max(len(features1), len(features2)) if max(len(features1), len(features2)) > 0 else 0
        return similarity
    except Exception as ex:
        logging.error(f"Error calculating similarity: {ex}")
        return 0  # Return a default value in case of an error

def extract_numeric_worm_features(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Extract numeric features of a file using pefile for worm detection.
    """
    res = {}
    try:
        # Reuse the numeric features extraction function for base data
        res.update(extract_numeric_features(file_path) or {})

    except Exception as ex:
        logging.error(f"An error occurred while processing {file_path}: {ex}", exc_info=True)

    return res

def check_worm_similarity(file_path, features_current):
    """
    Check similarity between the main file, collected files, and the current file for worm detection.
    """
    worm_detected = False

    try:
        # Compare with the main file if available and distinct from the current file
        if main_file_path and main_file_path != file_path:
            features_main = extract_numeric_worm_features(main_file_path)
            similarity_main = calculate_similarity_worm(features_current, features_main)
            if similarity_main > 0.86:
                logging.warning(
                    f"Main file '{main_file_path}' is potentially spreading the worm to '{file_path}' "
                    f"with similarity score: {similarity_main:.2f}"
                )
                worm_detected = True

        # Compare with each collected file in the file paths
        for collected_file_path in worm_file_paths:
            if collected_file_path != file_path:
                features_collected = extract_numeric_worm_features(collected_file_path)
                similarity_collected = calculate_similarity_worm(features_current, features_collected)
                if similarity_collected > 0.86:
                    logging.warning(
                        f"Worm has potentially spread to '{collected_file_path}' "
                        f"from '{file_path}' with similarity score: {similarity_collected:.2f}"
                    )
                    worm_detected = True

    except FileNotFoundError as fnf_error:
        logging.error(f"File not found: {fnf_error}")
    except Exception as ex:
        logging.error(f"An unexpected error occurred while checking worm similarity for '{file_path}': {ex}")

    return worm_detected

def worm_alert(file_path):

    if file_path in worm_alerted_files:
        logging.info(f"Worm alert already triggered for {file_path}, skipping...")
        return

    try:
        logging.info(f"Running worm detection for file '{file_path}'")

        # Extract features
        features_current = extract_numeric_worm_features(file_path)
        is_critical = file_path.startswith(main_drive_path) or file_path.startswith(system_root) or file_path.startswith(sandbox_system_root_directory)

        if is_critical:
            original_file_path = os.path.join(system_root, os.path.basename(file_path))
            sandbox_file_path = os.path.join(sandbox_system_root_directory, os.path.basename(file_path))

            if os.path.exists(original_file_path) and os.path.exists(sandbox_file_path):
                original_file_size = os.path.getsize(original_file_path)
                current_file_size = os.path.getsize(sandbox_file_path)
                size_difference = abs(current_file_size - original_file_size) / original_file_size

                original_file_mtime = os.path.getmtime(original_file_path)
                current_file_mtime = os.path.getmtime(sandbox_file_path)
                mtime_difference = abs(current_file_mtime - original_file_mtime)

                if size_difference > 0.10:
                    logging.warning(f"File size difference for '{file_path}' exceeds 10%.")
                    notify_user_worm(file_path, "HEUR:Win32.Worm.Critical.Agnostic.gen.Malware")
                    worm_alerted_files.append(file_path)
                    return  # Only flag once if a critical difference is found

                if mtime_difference > 3600:  # 3600 seconds = 1 hour
                    logging.warning(f"Modification time difference for '{file_path}' exceeds 1 hour.")
                    notify_user_worm(file_path, "HEUR:Win32.Worm.Critical.Time.Agnostic.gen.Malware")
                    worm_alerted_files.append(file_path)
                    return  # Only flag once if a critical difference is found

            # Proceed with worm detection based on critical file comparison
            worm_detected = check_worm_similarity(file_path, features_current)

            if worm_detected:
                logging.warning(f"Worm '{file_path}' detected in critical directory. Alerting user.")
                notify_user_worm(file_path, "HEUR:Win32.Worm.Classic.Critical.gen.Malware")
                worm_alerted_files.append(file_path)

        else:
            # Check for generic worm detection
            worm_detected = check_worm_similarity(file_path, features_current)
            worm_detected_count[file_path] = worm_detected_count.get(file_path, 0) + 1

            if worm_detected or worm_detected_count[file_path] >= 5:
                if file_path not in worm_alerted_files:
                    logging.warning(f"Worm '{file_path}' detected under 5 different names or as potential worm. Alerting user.")
                    notify_user_worm(file_path, "HEUR:Win32.Worm.Classic.gen.Malware")
                    worm_alerted_files.append(file_path)

                # Notify for all files that have reached the detection threshold
                for detected_file in worm_detected_count:
                    if worm_detected_count[detected_file] >= 5 and detected_file not in worm_alerted_files:
                        notify_user_worm(detected_file, "HEUR:Win32.Worm.Classic.gen.Malware")
                        worm_alerted_files.append(detected_file)

    except Exception as ex:
        logging.error(f"Error in worm detection for file {file_path}: {ex}")

def check_pe_file(file_path, signature_check, file_name):
    try:
        # Normalize the file path to lowercase for comparison
        normalized_path = os.path.abspath(file_path).lower()
        normalized_sandboxie = sandboxie_folder.lower()

        logging.info(f"File {file_path} is a valid PE file.")

        # Check if file is inside the Sandboxie folder
        if normalized_path.startswith(normalized_sandboxie):
            worm_alert(file_path)
            logging.info(f"File {file_path} is inside Sandboxie folder, scanned with worm_alert.")

        # Check for fake system files after signature validation
        if file_name in fake_system_files and os.path.abspath(file_path).startswith(main_drive_path):
            if not signature_check["is_valid"]:
                logging.warning(f"Detected fake system file: {file_path}")
                notify_user_for_detected_fake_system_file(file_path, file_name, "HEUR:Win32.FakeSystemFile.Dropper.gen")

    except Exception as ex:
        logging.error(f"Error checking PE file {file_path}: {ex}")

def is_zip_file(file_path):
    """
    Return True if file_path is a valid ZIP (AES or standard), False otherwise.
    """
    try:
        # Try standard ZIP
        with pyzipper.ZipFile(file_path, 'r'):
            return True
    except pyzipper.zipfile.BadZipFile:
        # Try AES ZIP
        try:
            with pyzipper.AESZipFile(file_path, 'r'):
                return True
        except Exception:
            return False
    except Exception as e:
        logging.error(f"Unexpected error checking ZIP: {e}")
        return False

def scan_file_real_time(file_path, signature_check, file_name, die_output, pe_file=False):
    """Scan file in real-time using multiple engines."""
    logging.info(f"Started scanning file: {file_path}")

    try:
        # Scan with Machine Learning AI for PE files
        try:
            if pe_file:
                is_malicious_machine_learning , malware_definition, benign_score = scan_file_with_machine_learning_ai(file_path)
                if is_malicious_machine_learning:
                    if benign_score < 0.93:
                        if signature_check["is_valid"]:
                            malware_definition = "SIG." + malware_definition
                        logging.warning(f"Infected file detected (ML): {file_path} - Virus: {malware_definition}")
                        return True, malware_definition, "ML"
                    elif benign_score >= 0.93:
                        logging.info(f"File is clean based on ML benign score: {file_path}")
                logging.info(f"No malware detected by Machine Learning in file: {file_path}")
        except Exception as ex:
            logging.error(f"An error occurred while scanning file with Machine Learning AI: {file_path}. Error: {ex}")

        # Worm analysis and fake file analysis
        try:
            if pe_file:
                check_pe_file(file_path, signature_check, file_name)
        except Exception as ex:
            logging.error(f"An error occurred while scanning the file for fake system files and worm analysis: {file_path}. Error: {ex}")

        # Scan with ClamAV
        try:
            result = scan_file_with_clamd(file_path)
            if result not in ("Clean", ""):
                if signature_check["is_valid"]:
                    result = "SIG." + result
                logging.warning(f"Infected file detected (ClamAV): {file_path} - Virus: {result}")
                return True, result, "ClamAV"
            logging.info(f"No malware detected by ClamAV in file: {file_path}")
        except Exception as ex:
            logging.error(f"An error occurred while scanning file with ClamAV: {file_path}. Error: {ex}")

        # Scan with YARA
        try:
            yara_result = scan_yara(file_path)
            if yara_result is not None and yara_result not in ("Clean", ""):
                if signature_check["is_valid"]:
                    yara_result = "SIG." + yara_result
                logging.warning(f"Infected file detected (YARA): {file_path} - Virus: {yara_result}")
                return True, yara_result, "YARA"
            logging.info(f"Scanned file with YARA: {file_path} - No viruses detected")
        except Exception as ex:
            logging.error(f"An error occurred while scanning file with YARA: {file_path}. Error: {ex}")

        # Scan TAR files
        try:
            if tarfile.is_tarfile(file_path):
                scan_result, virus_name = scan_tar_file(file_path)
                if scan_result and virus_name not in ("Clean", "F", "", [], None):
                    virus_str = str(virus_name) if virus_name else "Unknown"
                    if signature_check["is_valid"]:
                        virus_name = "SIG." + virus_str
                    logging.warning(f"Infected file detected (TAR): {file_path} - Virus: {virus_str}")
                    return True, virus_str, "TAR"
                logging.info(f"No malware detected in TAR file: {file_path}")
        except PermissionError:
            logging.error(f"Permission error occurred while scanning TAR file: {file_path}")
        except FileNotFoundError:
            logging.error(f"TAR file not found error occurred while scanning TAR file: {file_path}")
        except Exception as ex:
            logging.error(f"An error occurred while scanning TAR file: {file_path}. Error: {ex}")

        # Scan ZIP files
        try:
            if is_zip_file(file_path):
                scan_result, virus_name = scan_zip_file(file_path)
                if scan_result and virus_name not in ("Clean", ""):
                    if signature_check["is_valid"]:
                        virus_name = "SIG." + virus_name
                    logging.warning(f"Infected file detected (ZIP): {file_path} - Virus: {virus_name}")
                    return True, virus_name, "ZIP"
                logging.info(f"No malware detected in ZIP file: {file_path}")
        except PermissionError:
            logging.error(f"Permission error occurred while scanning ZIP file: {file_path}")
        except FileNotFoundError:
            logging.error(f"ZIP file not found error occurred while scanning ZIP file: {file_path}")
        except Exception as ex:
            logging.error(f"An error occurred while scanning ZIP file: {file_path}. Error: {ex}")

        # Scan 7z files
        try:
            if is_7z_file_from_output(die_output):
                scan_result, virus_name = scan_7z_file(file_path)
                if scan_result and virus_name not in ("Clean", ""):
                    if signature_check["is_valid"]:
                        virus_name = "SIG." + virus_name
                    logging.warning(f"Infected file detected (7z): {file_path} - Virus: {virus_name}")
                    return True, virus_name, "7z"
                logging.info(f"No malware detected in 7z file: {file_path}")
            else:
                logging.info(f"File is not a valid 7z archive: {file_path}")
        except PermissionError:
            logging.error(f"Permission error occurred while scanning 7Z file: {file_path}")
        except FileNotFoundError:
            logging.error(f"7Z file not found error occurred while scanning 7Z file: {file_path}")
        except Exception as ex:
            logging.error(f"An error occurred while scanning 7Z file: {file_path}. Error: {ex}")

    except Exception as ex:
        logging.error(f"An error occurred while scanning file: {file_path}. Error: {ex}")

    return False, "Clean", ""  # Default to clean if no malware found

# Read the file and store the names in a list (ignoring empty lines)
with open(system_file_names_path, "r") as f:
    fake_system_files = [line.strip() for line in f if line.strip()]

def activate_uefi_drive():
    # Check if the platform is Windows
    mount_command = 'mountvol X: /S'  # Command to mount UEFI drive
    try:
        # Execute the mountvol command
        subprocess.run(mount_command, shell=True, check=True, encoding="utf-8", errors="ignore")
        logging.info("UEFI drive activated!")
    except subprocess.CalledProcessError as ex:
        logging.error(f"Error mounting UEFI drive: {ex}")

threading.Thread(target=run_snort).start()
restart_clamd_thread()
clean_directories()
activate_uefi_drive() # Call the UEFI function
load_website_data()
load_antivirus_list()
# Load Antivirus and Microsoft digital signatures
antivirus_signatures = load_digital_signatures(digital_signatures_list_antivirus_path, "Antivirus digital signatures")
goodsign_signatures = load_digital_signatures(digital_signatures_list_antivirus_path, "UnHackMe digital signatures")

# Load ML definitions
try:
    with open(machine_learning_results_json, 'r') as results_file:
        ml_defs = json.load(results_file)
        malicious_numeric_features = ml_defs.get('malicious_numeric_features', [])
        malicious_file_names = ml_defs.get('malicious_file_names', [])
        benign_numeric_features = ml_defs.get('benign_numeric_features', [])
        benign_file_names = ml_defs.get('benign_file_names', [])
        logging.info("Machine Learning Definitions loaded!")
except Exception as ex:
    logging.error(f"Error loading ML definitions from {machine_learning_results_json}: {ex}")

try:
    # Load excluded rules from text file
    with open(excluded_rules_path, "r") as excluded_file:
        excluded_rules = excluded_file.read()
        logging.info("YARA Excluded Rules Definitions loaded!")
except Exception as ex:
    logging.error(f"Error loading excluded rules: {ex}")

try:
    # Load the precompiled yarGen rules from the .yrc file
    yarGen_rule = yara.load(yarGen_rule_path)
    logging.info("yarGen Rules Definitions loaded!")
except yara.Error as ex:
    logging.error(f"Error loading precompiled YARA rule: {ex}")

try:
    # Load the precompiled icewater rules from the .yrc file
    icewater_rule = yara.load(icewater_rule_path)
    logging.info("Icewater Rules Definitions loaded!")
except yara.Error as ex:
    logging.error(f"Error loading precompiled YARA rule: {ex}")

try:
    # Load the precompiled valhalla rules from the .yrc file
    valhalla_rule = yara.load(valhalla_rule_path)
    logging.info("Vallhalla Demo Rules Definitions loaded!")
except yara.Error as ex:
    logging.error(f"Error loading precompiled YARA rule: {ex}")

try:
    # Load the precompiled rules from the .yrc file
    compiled_rule = yara.load(compiled_rule_path)
    logging.info("YARA Rules Definitions loaded!")
except yara.Error as ex:
    logging.error(f"Error loading precompiled YARA rule: {ex}")

try:
    # Load the precompiled yaraxtr rule from the .yrc file using yara_x
    with open(yaraxtr_yrc_path, 'rb') as yara_x_f:
        yaraxtr_rule = yara_x.Rules.deserialize_from(yara_x_f)
    logging.info("YARA-X yaraxtr Rules Definitions loaded!")
except Exception as ex:
    logging.error(f"Error loading YARA-X rules: {ex}")

try:
    # Load the precompiled cx_freeze rule from the .yrc file using yara_x
    with open(cx_freeze_yrc_path, 'rb') as yara_x_cx_freeze:
        cx_freeze_rule = yara_x.Rules.deserialize_from(yara_x_cx_freeze)
    logging.info("YARA-X cx_freeze Rules Definitions loaded!")
except Exception as ex:
    logging.error(f"Error loading YARA-X rules: {ex}")

def has_known_extension(file_path):
    try:
        ext = os.path.splitext(file_path)[1].lower()
        logging.info(f"Extracted extension '{ext}' for file '{file_path}'")
        return ext in fileTypes
    except Exception as ex:
        logging.error(f"Error checking extension for file {file_path}: {ex}")
        return False

def is_readable(file_path):
    try:
        logging.info(f"Attempting to read file '{file_path}'")
        with open(file_path, 'r') as readable_file:
            file_data = readable_file.read(1024)
            if file_data:  # Check if file has readable content
                logging.info(f"File '{file_path}' is readable")
                return True
            return False
    except UnicodeDecodeError:
        logging.error(f"UnicodeDecodeError while reading file '{file_path}'")
        return False
    except Exception as ex:
        logging.error(f"Error reading file {file_path}: {ex}")
        return False

def is_ransomware(file_path):
    try:
        filename = os.path.basename(file_path)
        parts = filename.split('.')
        logging.info(f"Checking ransomware conditions for file '{file_path}' with parts '{parts}'")

        # Check if there are multiple extensions
        if len(parts) < 3:
            logging.info(f"File '{file_path}' does not have multiple extensions, not flagged as ransomware")
            return False

        # Check if the second last extension is known
        previous_extension = '.' + parts[-2].lower()
        if previous_extension not in fileTypes:
            logging.info(f"Previous extension '{previous_extension}' of file '{file_path}' is not known, not flagged as ransomware")
            return False

        # Check if the final extension is not in fileTypes
        final_extension = '.' + parts[-1].lower()
        if final_extension not in fileTypes:
            logging.warning(f"File '{file_path}' has unrecognized final extension '{final_extension}', checking if it might be ransomware sign")

            # Check if the file has a known extension or is readable
            if has_known_extension(file_path) or is_readable(file_path):
                logging.info(f"File '{file_path}' is not ransomware")
                return False
            else:
                logging.warning(f"File '{file_path}' might be a ransomware sign")
                return True

        logging.info(f"File '{file_path}' does not meet ransomware conditions")
        return False

    except Exception as ex:
        logging.error(f"Error checking ransomware for file {file_path}: {ex}")
        return False

def search_files_with_same_extension(directory, extension):
    try:
        logging.info(f"Searching for files with extension '{extension}' in directory '{directory}'")
        files_with_same_extension = []
        for root, _, files in os.walk(directory):
            for search_file in files:
                if search_file.endswith(extension):
                    files_with_same_extension.append(os.path.join(root, search_file))
        logging.info(f"Found {len(files_with_same_extension)} files with extension '{extension}'")
        return files_with_same_extension
    except Exception as ex:
        logging.error(f"Error searching for files with extension '{extension}' in directory '{directory}': {ex}")
        return []

def ransomware_alert(file_path):
    global ransomware_detection_count

    try:
        logging.info(f"Running ransomware alert check for file '{file_path}'")

        # Check the ransomware flag once.
        if is_ransomware(file_path):
            # If file is from the Sandboxie log folder, trigger Sandboxie-specific alert.
            if file_path.startswith(sandboxie_log_folder):
                ransomware_detection_count += 1
                logging.warning(f"File '{file_path}' (Sandboxie log) flagged as potential ransomware. Count: {ransomware_detection_count}")
                notify_user_ransomware(main_file_path, "HEUR:Win32.Ransom.Log.gen")
                logging.warning(f"User has been notified about potential ransomware in {main_file_path} (Sandboxie log alert)")

            # Normal processing for all flagged files.
            ransomware_detection_count += 1
            logging.warning(f"File '{file_path}' might be a ransomware sign. Count: {ransomware_detection_count}")

            # When exactly two alerts occur, search for files with the same extension.
            if ransomware_detection_count == 2:
                _, ext = os.path.splitext(file_path)
                if ext:
                    directory = os.path.dirname(file_path)
                    files_with_same_extension = search_files_with_same_extension(directory, ext)
                    for ransom_file in files_with_same_extension:
                        logging.info(f"Checking file '{ransom_file}' with same extension '{ext}'")
                        if is_ransomware(ransom_file):
                            logging.warning(f"File '{ransom_file}' might also be related to ransomware")

            # When detections reach a threshold, notify the user with a generic flag.
            if ransomware_detection_count >= 10:
                notify_user_ransomware(main_file_path, "HEUR:Win32.Ransom.gen")
                logging.warning(f"User has been notified about potential ransomware in {main_file_path}")

    except Exception as ex:
        logging.error(f"Error in ransomware_alert: {ex}")

executor = ThreadPoolExecutor(max_workers=1000)

def run_in_thread(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return executor.submit(fn, *args, **kwargs)
    return wrapper

# --- Main Scanning Function ---
@run_in_thread
def scan_and_warn(file_path,
                  mega_optimization_with_anti_false_positive=True,
                  command_flag=False,
                  flag_debloat=False,
                  flag_obfuscar=False,
                  flag_de4dot=False,
                  flag_fernflower=False,
                  nsis_flag=False,
                  ntdll_dropped=False):
    """
    Scans a file for potential issues.
    Only does ransomware_alert and worm_alert once per unique file path.
    """
    try:
        # Initialize variables
        perform_special_scan = False
        is_decompiled = False
        pe_file = False
        signature_check = {
            "has_microsoft_signature": False,
            "is_valid": False,
            "signature_status_issues": False
        }

        # Convert WindowsPath to string if necessary
        if isinstance(file_path, WindowsPath):
            file_path = str(file_path)

        # Ensure path is a string, exists, and is non-empty
        if not isinstance(file_path, str):
            logging.error(f"Invalid file_path type: {type(file_path).__name__}")
            return False

        # Ensure the file exists before proceeding.
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return False

        # Check if the file is empty.
        if os.path.getsize(file_path) == 0:
            logging.debug(f"File {file_path} is empty. Skipping scan.")
            return False

        # Normalize the original path
        norm_path = os.path.abspath(file_path)

        # Compute a quick MD5
        md5 = compute_md5(norm_path)

        # Initialize our seen-set once, on the function object
        if not hasattr(scan_and_warn, "_seen"):
            scan_and_warn._seen = set()

        # If we've already scanned this exact (path, hash), skip immediately
        key = (norm_path.lower(), md5)
        if key in scan_and_warn._seen:
            logging.debug(f"Skipping duplicate scan for {norm_path} (hash={md5})")
            return False

         # Mark it seen and proceed
        scan_and_warn._seen.add(key)

        # SNAPSHOT the cache entry _once_ up front:
        initial_md5_in_cache = file_md5_cache.get(norm_path)

        normalized_path = norm_path.lower()
        normalized_sandbox = os.path.abspath(sandboxie_folder).lower()
        normalized_de4dot = os.path.abspath(de4dot_sandboxie_dir).lower()

        # --- Route files based on origin folder ---
        if normalized_path.startswith(normalized_de4dot):
            perform_special_scan = True
            # Copy from de4dot sandbox to extracted directory and rescan
            dest = _copy_to_dest(norm_path, de4dot_extracted_dir)
            if dest is not None:
                scan_and_warn(dest,
                                mega_optimization_with_anti_false_positive,
                                command_flag,
                                flag_debloat,
                                flag_obfuscar,
                                flag_de4dot,
                                flag_fernflower,
                                nsis_flag,
                                ntdll_dropped)
        elif normalized_path.startswith(normalized_sandbox):
            # Check if this is a dropped ntdll.dll in the sandbox
            if normalized_path == sandboxed_ntdll_path:
                ntdll_dropped = True
                logging.warning(f"ntdll.dll dropped in sandbox at path: {normalized_path}")
                # Optionally force a special scan for this file
                perform_special_scan = True
                # You may choose a specific dir for ntdll analysis, or reuse existing staging dir
                dest = _copy_to_dest(norm_path, copied_sandbox_and_main_files_dir)
                if dest is not None:
                    scan_and_warn(
                        dest,
                        mega_optimization_with_anti_false_positive,
                        command_flag,
                        flag_debloat,
                        flag_obfuscar,
                        flag_de4dot,
                        flag_fernflower,
                        nsis_flag,
                        ntdll_dropped
                    )

            # --- General sandbox routing for other files ---
            perform_special_scan = True
            dest = _copy_to_dest(norm_path, copied_sandbox_and_main_files_dir)
            if dest is not None:
                scan_and_warn(
                    dest,
                    mega_optimization_with_anti_false_positive,
                    command_flag,
                    flag_debloat,
                    flag_obfuscar,
                    flag_de4dot,
                    flag_fernflower,
                    nsis_flag,
                    ntdll_dropped
                )

        # 1) Is this the first time we've seen this path?
        is_first_pass = norm_path not in file_md5_cache

        # Extract the file name
        file_name = os.path.basename(norm_path)

        # Try cache first
        if md5 in die_cache:
            die_output, plain_text_flag = die_cache[md5]
        else:
            die_output, plain_text_flag = get_die_output(norm_path)

        # Store for next time
        die_cache[md5] = (die_output, plain_text_flag)

        # Perform ransomware alert check
        if is_file_fully_unknown(die_output):
            if perform_special_scan:
                ransomware_alert(norm_path)
            if mega_optimization_with_anti_false_positive:
                logging.info(
                    f"Stopped analysis; unknown data detected in {norm_path}"
                )
                return False

        if is_advanced_installer_file_from_output(die_output):
            logging.info(f"File {norm_path} is a valid Advanced Installer file.")
            extracted_files = advanced_installer_extractor(file_path)
            for extracted_file in extracted_files:
                scan_and_warn(extracted_file)

        if is_pe_file_from_output(die_output):
            logging.info(f"File {norm_path} is a valid PE file.")
            pe_file = True

        if not is_first_pass and perform_special_scan and pe_file:
                worm_alert(norm_path)
                return True

        # On subsequent passes: skip if unchanged (unless forced)
        if initial_md5_in_cache == md5:
            logging.info(f"Skipping scan for unchanged file: {norm_path}")
            return False
        else:
            # File changed or forced: update MD5 and deep scan
            file_md5_cache[norm_path] = md5

        logging.info(f"Deep scanning file: {norm_path}")

        # Wrap norm_path in a Path once, up front
        wrap_norm_path = Path(norm_path)

        # Read raw binary data (for scanning, YARA, hashing, etc.)
        data_content = b""
        try:
            with open(norm_path, "rb") as f:
                data_content = f.read()
        except Exception as e:
            logging.error(f"Failed to read binary data from {norm_path}: {e}")

        # Read as UTF-8 text lines (for processing code/config/scripts/etc.)
        lines = []
        try:
            with open(norm_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            logging.error(f"Failed to read text lines from {norm_path}: {e}")

        # 1) Obfuscar-dir check
        if Path(obfuscar_dir) in wrap_norm_path.parents and not flag_obfuscar:
            flag_obfuscar = True
            logging.info(f"Flag set to True because '{norm_path}' is inside the Obfuscar directory '{obfuscar_dir}'.")

        # 2) de4dot directories check
        match = next(
            (Path(p) for p in (de4dot_extracted_dir, de4dot_sandboxie_dir)
            if Path(p) in wrap_norm_path.parents),
            None
        )
        if match and not flag_de4dot:
            flag_de4dot = True
            logging.info(
                f"Flag set to True because '{norm_path}' is inside the de4dot directory '{match}'"
        )

        # Check if the file content is valid non plain text data
        if not plain_text_flag:
            logging.info(f"File {norm_path} contains valid non plain text data.")
            # Attempt to extract the file
            try:
                logging.info(f"Attempting to extract file {norm_path}...")
                extracted_files = extract_all_files_with_7z(norm_path, nsis_flag)

                if extracted_files:
                    logging.info(f"Extraction successful for {norm_path}. Scanning extracted files...")
                    # Recursively scan each extracted file
                    for extracted_file in extracted_files:
                        logging.info(f"Scanning extracted file: {extracted_file}")
                        threading.Thread(target=scan_and_warn, args=(extracted_file,)).start()

                logging.info(f"File {norm_path} is not a valid archive or extraction failed. Proceeding with scanning.")
            except Exception as extraction_error:
                logging.error(f"Error during extraction of {norm_path}: {extraction_error}")

            if is_enigma1_virtual_box(die_output):
                extracted_path = try_unpack_enigma1(norm_path)
                if extracted_path:
                    logging.info(f"Unpack succeeded. Files are in: {extracted_path}")
                    threading.Thread(target=scan_and_warn, args=(extracted_path,)).start()
                else:
                    logging.info("Unpack failed for all known Enigma1 Virtual Box protected versions.")

            if is_packer_upx_output(die_output):
                upx_unpacked = extract_upx(norm_path)
                if upx_unpacked:
                    threading.Thread(target=scan_and_warn, args=(upx_unpacked,)).start()
                else:
                    logging.error(f"Failed to unpack {norm_path}")
            else:
                logging.info(f"Skipping non-UPX file: {norm_path}")

            if is_nsis_from_output(die_output):
                nsis_flag= True

            # Detect Inno Setup installer
            if is_inno_setup_archive_from_output(die_output):
                # Extract Inno Setup installer files
                extracted = extract_inno_setup(norm_path)
                if extracted is not None:
                    logging.info(f"Extracted {len(extracted)} files. Scanning...")
                    for inno_norm_path in extracted:
                        try:
                            # send to scan_and_warn for analysis
                            threading.Thread(target=scan_and_warn, args=(inno_norm_path,)).start()
                        except Exception as e:
                            logging.error(f"Error scanning {inno_norm_path}: {e}")
                else:
                    logging.error("Extraction failed; nothing to scan.")

            # Deobfuscate binaries obfuscated by Go Garble.
            if is_go_garble_from_output(die_output):
                # Generate output paths based on the file name and the specified directories
                output_path = os.path.join(ungarbler_dir, os.path.basename(norm_path))
                string_output_path = os.path.join(ungarbler_string_dir, os.path.basename(norm_path) + "_strings.txt")

                # Process the file and get the results
                results = process_file_go(norm_path, output_path, string_output_path)

                # Send the output files for scanning if they are created
                if results.get("patched_data"):
                    # Scan the patched binary file
                    threading.Thread(target=scan_and_warn, args=(output_path,)).start()

                if results.get("decrypt_func_list"):
                    # Scan the extracted strings file
                    threading.Thread(target=scan_and_warn, args=(string_output_path,)).start()
            # Check if it's a .pyc file and decompile via Pylingual
            if is_pyc_file_from_output(die_output):
                logging.info(
                    f"File {norm_path} is a .pyc (Python Compiled Module). Attempting Pylingual decompilation...")

                # 1) Decompile
                pylingual, pycdas = show_code_with_pylingual_pycdas(
                    file_path=norm_path,
                )

                # 2) Scan .py sources in-memory
                if pylingual:
                    logging.info("Scanning all decompiled .py files from Pylingual output.")
                    for fname, source in pylingual.items():
                        logging.info(f"Scheduling scan for decompiled file: {fname}")
                        threading.Thread(
                            target=scan_and_warn,
                            kwargs={"file_path": None, "content": source}
                        ).start()
                else:
                    logging.error(f"Pylingual decompilation failed for {norm_path}.")

                # 3) Scan non-.py resources in-memory
                if pycdas:
                    logging.info("Scanning all extracted resources from PyCDAS output.")
                    for rname, rcontent in pycdas.items():
                        logging.info(f"Scheduling scan for resource: {rname}")
                        threading.Thread(
                            target=scan_and_warn,
                            kwargs={"file_path": None, "content": rcontent}
                        ).start()
                else:
                    logging.info(f"No extra resources extracted for {norm_path}.")

            # Operation of the PE file
            if pe_file:
                logging.info(f"File {norm_path} is identified as a PE file.")

                # Perform signature check only if the file is non plain text data
                signature_check = check_signature(norm_path)
                logging.info(f"Signature check result for {norm_path}: {signature_check}")
                if not isinstance(signature_check, dict):
                    logging.error(f"check_signature did not return a dictionary for file: {norm_path}, received: {signature_check}")

                # Handle signature results
                if signature_check["has_microsoft_signature"]:
                    logging.info(f"Valid Microsoft signature detected for file: {norm_path}")
                    return False

                # Check for good digital signatures (valid_goodsign_signatures) and return false if they exist and are valid
                if signature_check.get("valid_goodsign_signatures"):
                    logging.info(f"Valid good signature(s) detected for file: {norm_path}: {signature_check['valid_goodsign_signatures']}")
                    return False

                if signature_check["is_valid"]:
                    logging.info(f"File '{norm_path}' has a valid signature. Skipping worm detection.")
                elif signature_check["signature_status_issues"] and not signature_check["no_signature"]:
                    logging.warning(f"File '{norm_path}' has signature issues. Proceeding with further checks.")
                    notify_user_invalid(norm_path, "Win32.Susp.InvalidSignature")

                # Detect .scr extension and trigger heuristic warning
                if norm_path.lower().endswith(".scr"):
                    logging.warning(f"Suspicious .scr file detected: {norm_path}")
                    notify_user_scr(norm_path, "HEUR:Win32.Susp.PE.SCR.gen")

                # Decompile the file in a separate thread
                decompile_thread = threading.Thread(target=decompile_file, args=(norm_path,))
                decompile_thread.start()

                # PE section extraction and scanning
                section_files = extract_pe_sections(norm_path)
                if section_files:
                    logging.info(f"Extracted {len(section_files)} PE sections. Scanning...")
                    for fpath in section_files:
                        try:
                            threading.Thread(target=scan_and_warn, args=(fpath,)).start()
                        except Exception as e:
                            logging.error(f"Error scanning PE section {fpath}: {e}")
                else:
                    logging.error("PE section extraction failed or no sections found.")

                # Extract resources
                extracted = extract_resources(norm_path, resource_extractor_dir)
                if extracted:
                    for file in extracted:
                        threading.Thread(target=scan_and_warn, args=(file,)).start()

                # Use the `debloat` library to optimize PE file for scanning
                try:
                    if not flag_debloat:
                        logging.info(f"Debloating PE file {norm_path} for faster scanning.")
                        optimized_norm_path = debloat_pe_file(norm_path)
                        if optimized_norm_path:
                            logging.info(f"Debloated file saved at: {optimized_norm_path}")
                            threading.Thread(
                                target=scan_and_warn,
                                args=(optimized_norm_path,),
                                kwargs={'flag_debloat': True}
                            ).start()
                        else:
                             logging.error(f"Debloating failed for {norm_path}, continuing with the original file.")
                except Exception as ex:
                    logging.error(f"Error during debloating of {norm_path}: {ex}")
        else:
            # If the file content is plain text, perform scanning with Meta Llama-3.2-1B
            logging.info(f"File {norm_path} does contain plain text data.")
            # Check if the norm_path equals the homepage change path.
            if norm_path == homepage_change_path:
                try:
                    for line in lines:
                        line = line.strip()
                        if line:
                            # Expecting a format like "Firefox,google.com"
                            parts = line.split(',')
                            if len(parts) == 2:
                                browser_tag, homepage_value = parts[0].strip(), parts[1].strip()
                                logging.info(
                                    f"Processing homepage change entry: Browser={browser_tag}, Homepage={homepage_value}")
                                # Call scan_code_for_links, using the homepage value as the code to scan.
                                # Pass the browser tag as the homepage_flag.
                                scan_code_for_links(homepage_value, norm_path, homepage_flag=browser_tag)
                            else:
                                logging.error(f"Invalid format in homepage change file: {line}")
                except Exception as ex:
                    logging.error(f"Error processing homepage change file {norm_path}: {ex}")

            # Log directory type based on file path
            log_directory_type(norm_path)

            # Check if the file is in decompiled_dir
            if norm_path.startswith(decompiled_dir):
                logging.info(f"File {norm_path} is in decompiled_dir.")
                is_decompiled = True

            source_dirs = [
                Path(decompiled_dir).resolve(),
                Path(FernFlower_decompiled_dir).resolve(),
                Path(dotnet_dir).resolve(),
                Path(nuitka_source_code_dir).resolve(),
            ]

            norm_path_resolved = Path(norm_path).resolve()
            ext = norm_path_resolved.suffix.lower()

            if ext in script_exts:
                try:
                    threading.Thread(
                        target=scan_file_with_meta_llama,
                        args=(norm_path,),
                    ).start()
                except Exception as ex:
                    logging.error(f"Error during scanning with Meta Llama-3.2-1B for file {norm_path}: {ex}")
            else:
                for src in source_dirs:
                    try:
                        norm_path_resolved.relative_to(src)
                    except ValueError:
                        continue
                    else:
                        try:
                            threading.Thread(
                                target=scan_file_with_meta_llama,
                                args=(norm_path,),
                            ).start()
                        except Exception as ex:
                            logging.error(
                                f"Error during scanning with Meta Llama-3.2-1B for file {norm_path}: {ex}"
                            )
                        break

            # Scan for malware in real-time only for plain text and command flag
            if command_flag:
                logging.info(f"Performing real-time malware detection for plain text file: {norm_path}...")
                real_time_scan_thread = threading.Thread(target=monitor_message.detect_malware, args=(norm_path,))
                real_time_scan_thread.start()

        # Check if the file is a known rootkit file
        if file_name in known_rootkit_files:
            logging.warning(f"Detected potential rootkit file: {norm_path}")
            rootkit_thread = threading.Thread(target=notify_user_for_detected_rootkit, args=(norm_path, f"HEUR:Rootkit.{file_name}"))
            rootkit_thread.start()

        # Process the file data including magic byte removal
        if not os.path.commonpath([norm_path, processed_dir]) == processed_dir:
            process_thread = threading.Thread(target=process_file_data, args=(norm_path, die_output))
            process_thread.start()

        # Check for fake file size
        if os.path.getsize(norm_path) > 100 * 1024 * 1024:  # File size > 100MB
            with open(norm_path, 'rb') as fake_file:
                file_content_read = fake_file.read(100 * 1024 * 1024)
                if file_content_read == b'\x00' * 100 * 1024 * 1024:  # 100MB of continuous `0x00` bytes
                    logging.warning(f"File {norm_path} is flagged as HEUR:FakeSize.gen")
                    fake_size = "HEUR:FakeSize.gen"
                    if signature_check and signature_check["is_valid"]:
                        fake_size = "HEUR:SIG.Win32.FakeSize.gen"
                    notify_user_fake_size_thread = threading.Thread(target=notify_user_fake_size, args=(norm_path, fake_size))
                    notify_user_fake_size_thread.start()

        # Perform real-time scan
        is_malicious, virus_names, engine_detected = scan_file_real_time(norm_path, signature_check, file_name, die_output, pe_file=pe_file)

        # Inside the scan check logic
        if is_malicious:
            # Concatenate multiple virus names into a single string without delimiters
            virus_name = ''.join(virus_names)
            logging.warning(f"File {norm_path} is malicious. Virus: {virus_name}")

            if virus_name.startswith("PUA."):
                notify_user_pua_thread = threading.Thread(target=notify_user_pua, args=(norm_path, virus_name, engine_detected))
                notify_user_pua_thread.start()
            else:
                notify_user_thread = threading.Thread(target=notify_user, args=(norm_path, virus_name, engine_detected))
                notify_user_thread.start()

        # Additional post-decompilation actions based on extracted file path
        if is_decompiled:
            logging.info(f"Checking original file path from decompiled data for: {norm_path}")
            original_norm_path_thread = threading.Thread(target=extract_original_norm_path_from_decompiled, args=(norm_path,))
            original_norm_path_thread.start()

        detection_result = detect_suspicious_filename_patterns(file_name, fileTypes)
        if detection_result['suspicious']:
            # Handle multiple attack types if present
            attack_types = []
            if detection_result['rlo_attack']:
                attack_types.append("RLO")
            if detection_result['excessive_spaces']:
                attack_types.append("Spaces")
            if detection_result['multiple_extensions']:
                attack_types.append("MultiExt")

            virus_name = f"HEUR:Susp.Name.{'+'.join(attack_types)}.gen"
            notify_user_susp_name(file_path, virus_name)

    except Exception as ex:
        logging.error(f"Error scanning file {norm_path}: {ex}")
        return False

def check_startup_directories():
    """Monitor startup directories for new files and handle them."""
    # Define the paths to check
    defaultbox_user_startup_folder = rf'{sandboxie_folder}\user\current\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup'
    defaultbox_programdata_startup_folder = rf'{sandboxie_folder}\ProgramData\Microsoft\Windows\Start Menu\Programs\Startup'

    # List of directories to check
    directories_to_check = [
        defaultbox_user_startup_folder,
        defaultbox_programdata_startup_folder
    ]

    # List to keep track of already alerted files
    alerted_files = []

    while True:
        try:
            for directory in directories_to_check:
                if os.path.exists(directory):
                    for file in os.listdir(directory):
                        file_path = os.path.join(directory, file)
                        if os.path.isfile(file_path) and file_path not in alerted_files:
                            die_output = get_die_output_binary(file_path)
                            if file_path.endswith('.wll') and is_pe_file_from_output(die_output):
                                malware_type = "HEUR:Win32.Startup.DLLwithWLL.gen.Malware"
                                message = f"Confirmed DLL malware detected: {file_path}\nVirus: {malware_type}"
                            ext = Path(file_path).suffix.lower()
                            if ext in script_exts:
                                malware_type = "HEUR:Win32.Startup.Script.gen.Malware"
                                message = f"Confirmed script malware detected: {file_path}\nVirus: {malware_type}"
                            elif file_path.endswith(('.dll', '.jar', '.msi', '.scr', '.hta',)):
                                malware_type = "HEUR:Win32.Startup.Susp.Extension.gen.Malware"
                                message = f"Confirmed malware with suspicious extension detected: {file_path}\nVirus: {malware_type}"
                            else:
                                malware_type = "HEUR:Win32.Startup.Susp.gen.Malware"
                                message = f"Suspicious startup file detected: {file_path}\nVirus: {malware_type}"

                            logging.warning(f"Suspicious or malicious startup file detected in {directory}: {file}")
                            notify_user_startup(file_path, message)
                            threading.Thread(target=scan_and_warn, args=(file_path,)).start()
                            alerted_files.append(file_path)
        except Exception as ex:
            logging.error(f"An error occurred while checking startup directories: {ex}")

def check_hosts_file_for_blocked_antivirus():
    try:
        if not os.path.exists(hosts_path):
            return False

        with open(hosts_path, 'r') as hosts_file:
            hosts_content = hosts_file.read()

        blocked_domains = []

        # Regular expression pattern to match domain or any subdomain
        domain_patterns = [re.escape(domain) + r'\b' for domain in antivirus_domains_data]
        pattern = r'\b(?:' + '|'.join(domain_patterns) + r')\b'

        # Find all matching domains/subdomains in hosts content
        matches = re.findall(pattern, hosts_content, flags=re.IGNORECASE)

        if matches:
            blocked_domains = list(set(matches))  # Remove duplicates

        if blocked_domains:
            logging.warning(f"Malicious hosts file detected: {hosts_path}")
            notify_user_hosts(hosts_path, "HEUR:Win32.Trojan.Hosts.Hijacker.DisableAV.gen")
            return True

    except Exception as ex:
        logging.error(f"Error reading hosts file: {ex}")

    return False

# Function to continuously monitor hosts file
def monitor_hosts_file():
    # Continuously check the hosts file
    while True:
        is_malicious_host = check_hosts_file_for_blocked_antivirus()

        if is_malicious_host:
            logging.info("Malicious hosts file detected and flagged.")
            break  # Stop monitoring after notifying once

def is_malicious_file(file_path, size_limit_kb):
    """ Check if the file is less than the given size limit """
    return os.path.getsize(file_path) < size_limit_kb * 1024

def check_uefi_directories():
    """ Continuously check the specified UEFI directories for malicious files """
    alerted_uefi_files = []
    known_uefi_files = list(set(uefi_100kb_paths + uefi_paths))  # Convert to list and ensure uniqueness

    while True:
        for uefi_path in uefi_paths + uefi_100kb_paths:
            if os.path.isfile(uefi_path) and uefi_path.endswith(".efi"):
                if uefi_path not in alerted_uefi_files:
                    if uefi_path in uefi_100kb_paths and is_malicious_file(uefi_path, 100):
                        logging.warning(f"Malicious file detected: {uefi_path}")
                        notify_user_uefi(uefi_path, "HEUR:Win32.UEFI.SecureBootRecovery.gen.Malware")
                        threading.Thread(target=scan_and_warn, args=(uefi_path,)).start()
                        alerted_uefi_files.append(uefi_path)
                    elif uefi_path in uefi_paths and is_malicious_file(uefi_path, 1024):
                        logging.warning(f"Malicious file detected: {uefi_path}")
                        notify_user_uefi(uefi_path, "HEUR:Win32.UEFI.ScreenLocker.Ransomware.gen.Malware")
                        threading.Thread(target=scan_and_warn, args=(uefi_path,)).start()
                        alerted_uefi_files.append(uefi_path)

        # Check for any new files in the EFI directory
        efi_dir = rf'{sandboxie_folder}\drive\X\EFI'
        for root, dirs, files in os.walk(efi_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if file_path.endswith(".efi") and file_path not in known_uefi_files and file_path not in alerted_uefi_files:
                    logging.warning(f"Unknown file detected: {file_path}")
                    notify_user_uefi(file_path, "HEUR:Win32.Bootkit.Startup.UEFI.gen.Malware")
                    threading.Thread(target=scan_and_warn, args=(file_path,)).start()
                    alerted_uefi_files.append(file_path)

if __name__ == "__main__":
    main()
