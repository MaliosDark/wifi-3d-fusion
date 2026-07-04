#!/usr/bin/env python3
import argparse, os, numpy as np
from pathlib import Path

def gen_person_seq(T, D, rng, freq_bias, phase_bias, motion_prob=0.3):
    amp_base = np.clip(rng.normal(1.0 + 0.2*freq_bias, 0.05, size=D), 0.5, 2.0)
    x = np.linspace(0, 1, D, dtype=np.float32)
    phase_base = 2.0*np.pi*(phase_bias*x + 0.05*rng.normal(size=D)).astype(np.float32)
    seq = np.zeros((T, D, 2), dtype=np.float32)
    for t in range(T):
        moving = (rng.random() < motion_prob)
        amp = amp_base * (1.0 + (0.03 if moving else 0.01)*np.sin(2*np.pi*(t/T)*(1.0+0.2*freq_bias)))
        amp += rng.normal(0, 0.01, size=D)
        phase = phase_base + 0.3*np.sin(2*np.pi*(t/T)*(1.5+phase_bias)) + rng.normal(0,0.02,size=D)
        seq[t,:,0] = amp
        seq[t,:,1] = phase
    return seq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="data/reid")
    ap.add_argument("--num_persons", type=int, default=3)
    ap.add_argument("--seqs_per_person", type=int, default=40)
    ap.add_argument("--seq_secs", type=float, default=2.0)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--subcarriers", type=int, default=128)
    ap.add_argument("--multi_ap", action="store_true")
    ap.add_argument("--aps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)  # <-- instanciado

    T = int(args.seq_secs * args.fps)
    D = args.subcarriers
    Path(args.out_root).mkdir(parents=True, exist_ok=True)

    for pid in range(args.num_persons):
        pdir = Path(args.out_root)/f"person_{pid:03d}"
        pdir.mkdir(parents=True, exist_ok=True)
        if args.multi_ap:
            for a in range(args.aps):
                (pdir/f"ap_{a}").mkdir(parents=True, exist_ok=True)

        freq_bias = rng.uniform(-0.5, 0.5)
        phase_bias = rng.uniform(-0.5, 0.5)

        for k in range(args.seqs_per_person):
            if args.multi_ap:
                for a in range(args.aps):
                    fb = freq_bias + 0.1*a
                    pb = phase_bias + 0.07*a
                    seq = gen_person_seq(T, D, rng, fb, pb, motion_prob=0.4)
                    np.savez_compressed(pdir/f"ap_{a}"/f"seq_{k:03d}.npz", seq=seq)
            else:
                seq = gen_person_seq(T, D, rng, freq_bias, phase_bias, motion_prob=0.4)
                np.savez_compressed(pdir/f"seq_{k:03d}.npz", seq=seq)

    print(f"Done. Persons={args.num_persons} seqs/person={args.seqs_per_person} T={T} D={D} multi_ap={args.multi_ap}")

if __name__ == "__main__":
    main()
