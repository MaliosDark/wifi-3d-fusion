import argparse, os, time, numpy as np
from loguru import logger
from src.preprocess.csi import build_sequence

def get_source(source: str, port: int, mtu: int, iface: str):
    if source == "esp32":
        from src.csi_sources.esp32_udp import ESP32UDPCSISource as Src
        return Src(port=port, mtu=mtu)
    elif source == "nexmon":
        from src.csi_sources.monitor_radiotap import MonitorRadiotapSource as Src
        return Src(iface=iface or "wlan0")
    else:
        raise ValueError("source must be 'esp32' or 'nexmon'.")

def main(a):
    os.makedirs(a.out_root, exist_ok=True)
    person_dir = os.path.join(a.out_root, f"person_{int(a.person):03d}")
    os.makedirs(person_dir, exist_ok=True)

    src = get_source(a.source, a.port, a.mtu, a.iface)
    buf = []
    seq_frames = int(a.seq_secs * a.fps)
    last_save = time.time()

    logger.info(f"Recording: person={a.person} source={a.source} -> {person_dir}")
    for ts, vec in src.frames():
        if np.iscomplexobj(vec):
            csi = vec
        else:
            csi = vec.astype(np.float32).astype(np.complex64)
        buf.append(csi)
        if len(buf) > seq_frames:
            buf.pop(0)

        if len(buf) == seq_frames and (time.time() - last_save) >= a.cooldown:
            seq = build_sequence(buf)
            idx = len([f for f in os.listdir(person_dir) if f.endswith(".npz")])
            out = os.path.join(person_dir, f"seq_{idx:03d}.npz")
            np.savez_compressed(out, seq=seq)
            logger.info(f"Saved {out}  shape={seq.shape}")
            last_save = time.time()
            if a.max_sequences and idx+1 >= a.max_sequences:
                logger.info("Reached max sequences; stopping.")
                break

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["esp32","nexmon"], default="esp32")
    ap.add_argument("--iface", default="", help="monitor interface for nexmon")
    ap.add_argument("--port", type=int, default=5566)
    ap.add_argument("--mtu",  type=int, default=2000)
    ap.add_argument("--out_root", default="data/reid")
    ap.add_argument("--person", required=True, type=int)
    ap.add_argument("--seq_secs", type=float, default=2.0)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--cooldown", type=float, default=1.0)
    ap.add_argument("--max_sequences", type=int, default=0, help="0=unlimited")
    a = ap.parse_args()
    main(a)
