from scapy.all import AsyncSniffer
from loguru import logger
import numpy as np, time

class MonitorRadiotapSource:
    """
    Sniffea 802.11 en modo monitor y extrae RSSI (Radiotap.dBm_AntSignal).
    Devuelve (timestamp, vector amplitud) tamaño 256 normalizado a [0,1].
    """
    def __init__(self, iface="wlan0"):
        self.iface = iface
        self.N = 256

    def frames(self):
        q = []
        out = []

        def onpkt(pkt):
            # Radiotap RSSI
            rssi = getattr(pkt, 'dBm_AntSignal', None)
            if rssi is None:
                try:
                    rssi = pkt[0].dBm_AntSignal
                except Exception:
                    return
            q.append((time.time(), float(rssi)))
            if len(q) >= self.N:
                ts = q[-1][0]
                arr = np.array([x[1] for x in q], dtype=np.float32); q.clear()
                amp = (arr + 100.0) / 80.0   # mapea dBm [-100,-20] → [0,1]
                out.append((ts, amp))

        sniffer = AsyncSniffer(iface=self.iface, store=False, prn=onpkt, monitor=True)
        sniffer.start()
        logger.info(f"[monitor] sniffing on {self.iface}")
        try:
            while True:
                if out:
                    yield out.pop(0)
                else:
                    time.sleep(0.01)
        finally:
            sniffer.stop()
