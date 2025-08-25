import os, glob
import numpy as np
from torch.utils.data import Dataset

def _load_npz_multi(path:str):
    """Load either single-AP (seq) or multi-AP (seq_ap0, seq_ap1, ...)."""
    obj = np.load(path)
    if 'seq' in obj.files:
        return [obj['seq'].astype(np.float32)]
    # collect keys seq_ap*
    seqs = []
    for k in sorted([k for k in obj.files if k.startswith('seq_ap')]):
        seqs.append(obj[k].astype(np.float32))
    if not seqs:
        raise ValueError(f"No seq/seq_ap* in {path}")
    return seqs

class CSIMultiAPDataset(Dataset):
    """
    Layout 1 (flat):
      data_root/
        person_000/ seq_000.npz (seq or seq_ap0/1/..), ...
        person_001/ ...
    Layout 2 (per-AP subdirs):
      data_root/
        person_000/
          ap_0/ seq_000.npz ...
          ap_1/ seq_000.npz ...
    Returns:
      if multi_ap: list of (T,D,2) seqs per AP
      else: single (T,D,2)
    """
    def __init__(self, data_root:str, split_txt:str|None=None, min_aps:int=1, train_aug=False):
        self.items:list[str] = []
        self.labels:list[int] = []
        self.min_aps = min_aps
        self.train_aug = train_aug
        root = data_root
        if split_txt and os.path.exists(split_txt):
            with open(split_txt) as f:
                rels = [l.strip() for l in f if l.strip()]
            for rel in rels: self.items.append(os.path.join(root, rel))
        else:
            for pid_dir in sorted(glob.glob(os.path.join(root, "person_*"))):
                # flat npz
                npzs = sorted(glob.glob(os.path.join(pid_dir, "*.npz")))
                if npzs:
                    self.items += npzs
                # per-AP
                ap_dirs = sorted(glob.glob(os.path.join(pid_dir, "ap_*")))
                for apd in ap_dirs:
                    self.items += sorted(glob.glob(os.path.join(apd, "*.npz")))
        # Build labels
        for p in self.items:
            pid = os.path.basename(os.path.dirname(p))
            # if in ap_x/, parent is person_x
            if pid.startswith("ap_"):
                pid = os.path.basename(os.path.dirname(os.path.dirname(p)))
            self.labels.append(int(pid.split("_")[-1]))

    def __len__(self): return len(self.items)

    def _gather_multiap_for_file(self, path:str):
        # If file contains multi-ap dict -> return that
        seqs = _load_npz_multi(path)
        if len(seqs) >= self.min_aps:
            return seqs
        # If not, try to find siblings for same seq id across ap_* dirs
        base = os.path.basename(path)
        person_dir = os.path.dirname(path)
        if os.path.basename(person_dir).startswith("ap_"):
            # collect same base from all ap_* dirs
            root = os.path.dirname(person_dir)
            ap_dirs = sorted(glob.glob(os.path.join(root, "ap_*")))
            seqs2 = []
            for apd in ap_dirs:
                cand = os.path.join(apd, base)
                if os.path.exists(cand):
                    s = _load_npz_multi(cand)
                    seqs2 += s
            seqs = seqs2 if seqs2 else seqs
        return seqs

    def __getitem__(self, i):
        path = self.items[i]
        seqs = self._gather_multiap_for_file(path)
        y = self.labels[i]
        # Normalize shapes & trim D to min across APs
        T = min(s.shape[0] for s in seqs)
        D = min(s.shape[1] for s in seqs)
        seqs = [s[:T,:D,:] for s in seqs]
        return seqs, y
