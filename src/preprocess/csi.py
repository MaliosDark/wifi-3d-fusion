import numpy as np

def _detrend_phase(phi: np.ndarray) -> np.ndarray:
    """Remove linear trend in phase to mitigate CFO/SFO."""
    x = np.arange(phi.size, dtype=np.float32)
    k, b = np.polyfit(x, phi, 1)
    return phi - (k * x + b)

def csi_frame_to_feat(csi_vec: np.ndarray) -> np.ndarray:
    """
    Convert one complex CSI frame -> (D,2) features: [amp_norm, phase_norm].
    Accepts 1D complex vector (flattened).
    """
    if np.iscomplexobj(csi_vec):
        comp = csi_vec.astype(np.complex64, copy=False)
    else:
        # Amplitude-only input -> treat as real part of complex signal
        comp = csi_vec.astype(np.float32, copy=False).astype(np.complex64)
    amp = np.abs(comp).astype(np.float32)
    ph  = np.angle(comp).astype(np.float32)
    ph  = np.unwrap(ph)
    ph  = _detrend_phase(ph)

    def z(x: np.ndarray) -> np.ndarray:
        s = x.std() + 1e-6
        return (x - x.mean()) / s

    amp = z(amp)
    ph  = z(ph)
    return np.stack([amp, ph], axis=-1)

def build_sequence(window: list[np.ndarray]) -> np.ndarray:
    """
    window: list of complex CSI frames (1D arrays).
    Returns (T, D, 2) stacked features.
    """
    feats = [csi_frame_to_feat(x) for x in window]
    D = min(f.shape[0] for f in feats)
    feats = [f[:D] for f in feats]
    return np.stack(feats, axis=0)
