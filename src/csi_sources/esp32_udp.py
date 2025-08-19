import socket, json, time, threading
from typing import Iterator, Dict, Any, Optional, Tuple, List
import numpy as np
from loguru import logger

def _parse_esp32_json(pkt: bytes) -> Optional[np.ndarray]:
    """Parse ESP32-CSI UDP JSON (e.g., ESP32-CSI-Tool variants).
       Expect keys like: type:'CSI_DATA', mac, rssi, len, csi (list of ints).
       Returns complex CSI array shape (Nsub,) as complex64."""
    try:
        obj = json.loads(pkt.decode('utf-8', errors='ignore'))
        if isinstance(obj, dict) and obj.get('type','').upper().startswith('CSI'):
            csi = obj.get('csi') or obj.get('csi_data') or obj.get('data')
            if csi is None: return None
            csi = np.asarray(csi, dtype=np.int16)
            # ESP32 typically interleaves I/Q; convert if needed
            if csi.ndim == 1 and (csi.size % 2) == 0:
                iq = csi.astype(np.float32).reshape(-1,2)
                comp = iq[:,0] + 1j*iq[:,1]
                return comp.astype(np.complex64)
            # Or already amplitude/phase? Try to interpret as complex
            if np.iscomplexobj(csi): return csi.astype(np.complex64)
            return csi.astype(np.complex64)
        return None
    except Exception:
        return None

class ESP32UDPCSISource:
    """Listen for ESP32-CSI on UDP and yield (ts, csi_vector)."""
    def __init__(self, port: int = 5566, mtu: int = 2000, bind: str = "0.0.0.0"):
        self.port = port
        self.mtu = mtu
        self.bind = bind
        self._sock = None
        self._stop = threading.Event()

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.bind, self.port))
        logger.info(f"[ESP32] UDP listening on {self.bind}:{self.port}")

    def stop(self):
        self._stop.set()
        if self._sock:
            try: self._sock.close()
            except: pass

    def frames(self) -> Iterator[Tuple[float, np.ndarray]]:
        if self._sock is None: self.start()
        while not self._stop.is_set():
            data, _ = self._sock.recvfrom(self.mtu)
            arr = _parse_esp32_json(data)
            if arr is None: continue
            yield (time.time(), arr)
