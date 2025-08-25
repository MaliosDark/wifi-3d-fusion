_SCAPY_OK = False  # Force PyShark fallback for better compatibility
try:
    from scapy.all import AsyncSniffer
except ImportError:
    pass
from loguru import logger
import numpy as np, time


class MonitorRadiotapSource:
    """
    Sniffs 802.11 packets in monitor mode and extracts RSSI (Radiotap.dBm_AntSignal).
    Returns (timestamp, amplitude_vector) of configurable batch size, normalized to [0,1].
    """
    def __init__(self, iface="mon0", batch_size=256):
        self.iface = iface
        self.N = batch_size

    def frames(self):
        if _SCAPY_OK:
            q = []
            out = []

            def onpkt(pkt):
                rssi = getattr(pkt, 'dBm_AntSignal', None)
                if rssi is None:
                    try:
                        rssi = pkt[0].dBm_AntSignal
                    except Exception as e:
                        logger.warning(f"Packet missing dBm_AntSignal: {e}")
                        return
                q.append((time.time(), float(rssi)))
                if len(q) >= self.N:
                    ts = q[-1][0]
                    arr = np.array([x[1] for x in q], dtype=np.float32)
                    q.clear()
                    amp = (arr + 100.0) / 80.0   # map dBm [-100,-20] → [0,1]
                    out.append((ts, amp))

            sniffer = AsyncSniffer(iface=self.iface, store=False, prn=onpkt, monitor=True)
            sniffer.start()
            logger.info(f"[monitor] sniffing on {self.iface} with batch size {self.N}")
            try:
                last_packet_time = time.time()
                while True:
                    if out:
                        yield out.pop(0)
                        last_packet_time = time.time()
                    else:
                        time.sleep(0.01)
                        # Warn if no packets for 10 seconds
                        if time.time() - last_packet_time > 10:
                            logger.warning(f"No packets received on {self.iface} for 10 seconds. Check traffic and interface mode.")
                            last_packet_time = time.time()
            finally:
                sniffer.stop()
        else:
            # PyShark fallback
            import pyshark
            logger.info(f"[monitor][pyshark] sniffing on {self.iface} with batch size {self.N}")
            cap = pyshark.LiveCapture(interface=self.iface)
            q = []
            last_packet_time = time.time()
            for pkt in cap.sniff_continuously():
                try:
                    logger.debug(f"PyShark packet: {str(pkt)}")
                    rssi = None
                    # Try to extract RSSI from Radiotap
                    if hasattr(pkt, 'radiotap') and hasattr(pkt.radiotap, 'dbm_antsignal'):
                        rssi = float(pkt.radiotap.dbm_antsignal)
                        logger.debug(f"PyShark dbm_antsignal: {rssi}")
                    if rssi is not None:
                        q.append((time.time(), rssi))
                        if len(q) >= self.N:
                            ts = q[-1][0]
                            arr = np.array([x[1] for x in q], dtype=np.float32)
                            q.clear()
                            amp = (arr + 100.0) / 80.0
                            logger.info(f"Yielding batch with mean RSSI: {amp.mean():.2f}, std: {amp.std():.2f}")
                            yield (ts, amp)
                        last_packet_time = time.time()
                    else:
                        logger.warning("PyShark packet missing dbm_antsignal")
                except Exception as e:
                    logger.warning(f"PyShark error: {e}")
                # Warn if no packets for 10 seconds
                if time.time() - last_packet_time > 10:
                    logger.warning(f"No packets received on {self.iface} for 10 seconds. Check traffic and interface mode.")
                    last_packet_time = time.time()
