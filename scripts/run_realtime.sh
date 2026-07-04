#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate || true

SOURCE="esp32"
IFACE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2;;
    --iface)  IFACE="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

python - <<PY
import threading, time, sys, numpy as np
from loguru import logger
from src.common.config import load_cfg, ensure_dirs
from src.pipeline.realtime_detector import MovementDetector
from src.pipeline.realtime_viewer import LivePointCloud

cfg = load_cfg(); ensure_dirs()

source = "${SOURCE}" or cfg.get('source','esp32')

if source == 'esp32':
    from src.csi_sources.esp32_udp import ESP32UDPCSISource as Src
    src = Src(port=cfg['esp32_udp_port'], mtu=cfg['esp32_mtu'])
elif source == 'nexmon':
    from src.csi_sources.nexmon_pcap import NexmonPCAPSource as Src
    src = Src(iface=cfg['nexmon_iface'], pcap_path=cfg['nexmon_pcap'], tcpdump_filter=cfg['tcpdump_filter'])
else:
    from src.csi_sources.monitor_radiotap import MonitorRadiotapSource as Src
    iface = "${IFACE}" or cfg.get('nexmon_iface','wlan0')
    src = Src(iface=iface)

det = MovementDetector(win_seconds=cfg['win_seconds'], threshold=cfg['movement_threshold'], debounce=cfg['debounce_seconds'])
viewer = LivePointCloud()
th = threading.Thread(target=viewer.run, daemon=True); th.start()

for ts, vec in src.frames():
    amp = np.asarray(vec, dtype=np.float32)
    viewer.update_from_csi(amp)
    evt = det.update(ts, amp)
    if evt: logger.info(f"EVENT: {evt}")
PY
