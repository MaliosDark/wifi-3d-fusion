import numpy as np, time, torch
from collections import deque
from src.preprocess.csi import build_sequence
from src.models.who_encoder import WhoFiEncoder

class RealtimeIdentifier:
    def __init__(self, feat_dim: int, ckpt_path: str, seq_secs: float = 2.0, fps: float = 20.0, smooth: float = 0.8, device: str | None = None):
        print(f"[DEBUG] RealtimeIdentifier init: feat_dim={feat_dim}, ckpt_path={ckpt_path}")
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        import os
        try:
            print(f"[DEBUG] Loading checkpoint from {ckpt_path}...")
            if not os.path.isfile(ckpt_path):
                print(f"[ERROR] Checkpoint file does not exist: {ckpt_path}")
                self.model = None
                return
            ckpt = torch.load(ckpt_path, map_location='cpu')
            print(f"[DEBUG] Loaded checkpoint keys: {list(ckpt.keys())}")
            ckpt_feat_dim = ckpt.get('feat_dim', feat_dim)
            ckpt_num_ids = ckpt.get('num_ids', 100)
            print(f"[DEBUG] feat_dim from checkpoint: {ckpt_feat_dim}, num_ids: {ckpt_num_ids}")
            if 'model' not in ckpt:
                print(f"[ERROR] Checkpoint missing 'model' key: {ckpt_path}")
                self.model = None
                return
            self.model = WhoFiEncoder(ckpt_feat_dim, ckpt_num_ids).to(self.device)
            print(f"[DEBUG] Model created: WhoFiEncoder(feat_dim={ckpt_feat_dim}, num_ids={ckpt_num_ids})")
            self.model.load_state_dict(ckpt['model'], strict=False)
            print(f"[DEBUG] State dict loaded into model.")
            self.model.eval()
            print(f"[DEBUG] Model set to eval mode.")
        except Exception as e:
            print(f"[ERROR] Failed to load model from {ckpt_path}: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
        self.buf = deque()
        self.maxlen = int(seq_secs * fps)
        self.gallery: dict[int, np.ndarray] = {}
        self.smooth = smooth
        self.last_pred: tuple[int|None, float] | None = None

    def push(self, ts: float, csi_vec: np.ndarray):
        self.buf.append(csi_vec)
        if len(self.buf) > self.maxlen:
            self.buf.popleft()

    def _embed(self, seq: np.ndarray) -> np.ndarray:
        x = seq.reshape(1, seq.shape[0], seq.shape[1]*seq.shape[2]).astype(np.float32)
        xt = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            _, emb = self.model(xt)
        return emb[0].cpu().numpy()

    def enroll(self, pid: int, shots: int = 10) -> bool:
        if len(self.buf) < self.maxlen: return False
        feats = []
        frames = list(self.buf)[-self.maxlen:]
        for _ in range(shots):
            seq = build_sequence(frames)
            feats.append(self._embed(seq))
        proto = np.mean(np.stack(feats,0), axis=0)
        proto /= (np.linalg.norm(proto)+1e-9)
        self.gallery[pid] = proto
        return True

    def infer(self):
        if len(self.buf) < self.maxlen or not self.gallery:
            return None
        seq = build_sequence(list(self.buf)[-self.maxlen:])
        f = self._embed(seq)
        f = f / (np.linalg.norm(f)+1e-9)
        best_pid, best_sim = None, -1.0
        for pid, proto in self.gallery.items():
            s = float(f @ proto)
            if s > best_sim: best_sim, best_pid = s, pid
        if self.last_pred is None:
            self.last_pred = (best_pid, best_sim)
        else:
            pid0, sim0 = self.last_pred
            if best_pid == pid0:
                best_sim = self.smooth*sim0 + (1-self.smooth)*best_sim
                self.last_pred = (best_pid, best_sim)
            else:
                if best_sim > sim0 + 0.05:
                    self.last_pred = (best_pid, best_sim)
        pid, sim = self.last_pred
        return {"pid": pid, "score": sim, "t": time.time()}
