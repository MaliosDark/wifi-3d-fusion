import numpy as np
rng = np.random.default_rng

def phase_jitter(seq, max_std=0.15):
    # seq: (T,D,2), channel 1 is phase
    out = seq.copy()
    std = rng.uniform(0.0, max_std)
    noise = rng.normal(0.0, std, size=out[...,1].shape).astype(np.float32)
    out[...,1] = out[...,1] + noise
    return out

def time_mask(seq, max_masks=2, max_frac=0.1):
    T = seq.shape[0]
    out = seq.copy()
    for _ in range(rng.integers(0, max_masks+1)):
        w = max(1, int(T * rng.uniform(0.02, max_frac)))
        s = rng.integers(0, max(1, T-w+1))
        out[s:s+w] = 0.0
    return out

def freq_mask(seq, max_masks=2, max_frac=0.1):
    # mask along D dimension
    D = seq.shape[1]
    out = seq.copy()
    for _ in range(rng.integers(0, max_masks+1)):
        w = max(1, int(D * rng.uniform(0.02, max_frac)))
        s = rng.integers(0, max(1, D-w+1))
        out[:, s:s+w, :] = 0.0
    return out

def random_augment(seq, p_phase=0.7, p_time=0.5, p_freq=0.5):
    out = seq
    if rng.random() < p_phase: out = phase_jitter(out)
    if rng.random() < p_time:  out = time_mask(out)
    if rng.random() < p_freq:  out = freq_mask(out)
    return out
