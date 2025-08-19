import numpy as np, time, collections
from loguru import logger

class MovementDetector:
    def __init__(self, win_seconds=2.0, threshold=0.08, debounce=1.0):
        self.win = win_seconds
        self.thr = threshold
        self.debounce = debounce
        self.last_trigger = 0.0
        self.buf = collections.deque()

    def update(self, ts: float, csi_vec: np.ndarray):
        # Use amplitude variance over the window as a robust motion proxy.
        amp = np.abs(csi_vec).astype(np.float32)
        self.buf.append((ts, amp))
        # Drop old
        while self.buf and (ts - self.buf[0][0] > self.win):
            self.buf.popleft()
        # Compute variance across time for each subcarrier, then mean
        if len(self.buf) < 3: return None
        A = np.stack([a for _, a in self.buf], axis=0)
        score = float(np.mean(np.var(A, axis=0)))
        now = time.time()
        if score >= self.thr and (now - self.last_trigger) > self.debounce:
            self.last_trigger = now
            logger.info(f"[detect] movement score={score:.4f}")
            return {"event":"movement", "score": score, "t": ts}
        return None
