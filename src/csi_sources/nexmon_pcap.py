import subprocess, os, time, io
from pathlib import Path
from typing import Iterator, Tuple
import numpy as np
from loguru import logger
import csiread

class NexmonPCAPSource:
    """
    Captures packets using tcpdump to a rolling pcap and parses CSI with csiread.
    Requires: nexmon_csi firmware on capture device + monitor mode iface.
    """
    def __init__(self, iface: str = "wlan0", pcap_path: str = "env/nexmon_live.pcap",
                 tcpdump_filter: str = "type data"):
        self.iface = iface
        self.pcap_path = Path(pcap_path)
        self.tcpdump_filter = tcpdump_filter
        self._proc = None

    def start(self):
        self.pcap_path.parent.mkdir(parents=True, exist_ok=True)
        if self.pcap_path.exists():
            self.pcap_path.unlink()
        cmd = ["tcpdump", "-i", self.iface, "-s", "0", "-U", "-w", str(self.pcap_path), self.tcpdump_filter]
        logger.info(f"[Nexmon] starting tcpdump: {' '.join(cmd)}")
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setpgrp)

    def stop(self):
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass

    def frames(self) -> Iterator[Tuple[float, np.ndarray]]:
        self.start()
        last_size = 0
        try:
            while True:
                if not self.pcap_path.exists():
                    time.sleep(0.1); continue
                sz = self.pcap_path.stat().st_size
                if sz <= last_size:
                    time.sleep(0.1); continue
                last_size = sz
                # Parse whole file (csiread expects file path); yields latest CSIs
                try:
                    csidata = csiread.Nexmon(str(self.pcap_path), chip='4358', iqswap=True, enhance=True)
                    csidata.read()
                    for csi, ts in zip(csidata.csi, csidata.timestamp):
                        # csi is shape (Nsub, Nrx, Ntx) as complex
                        arr = np.asarray(csi).astype(np.complex64)
                        yield (float(ts), arr.ravel())
                except Exception as e:
                    logger.warning(f"[Nexmon] parse error: {e}")
                time.sleep(0.2)
        finally:
            self.stop()
