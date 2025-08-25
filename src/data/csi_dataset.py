import os, glob
import numpy as np
from torch.utils.data import Dataset

class CSIDataset(Dataset):
    """
    Expected layout:
      data_root/
        person_000/
          seq_000.npz   # contains: seq (T,D,2)
          seq_001.npz
        person_001/
          ...
    Optionally use split list files with relative paths.
    """
    def __init__(self, data_root: str, split_txt: str | None = None):
        self.items: list[str] = []
        self.labels: list[int] = []
        root = data_root
        if split_txt and os.path.exists(split_txt):
            with open(split_txt) as f:
                rels = [l.strip() for l in f if l.strip()]
            for rel in rels:
                p = os.path.join(root, rel)
                self.items.append(p)
        else:
            for pid_dir in sorted(glob.glob(os.path.join(root, "person_*"))):
                for f in sorted(glob.glob(os.path.join(pid_dir, "*.npz"))):
                    self.items.append(f)
        for p in self.items:
            pid = os.path.basename(os.path.dirname(p))
            self.labels.append(int(pid.split("_")[-1]))

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        obj = np.load(self.items[i])
        seq = obj["seq"].astype(np.float32)  # (T,D,2)
        T, D, C = seq.shape
        x = seq.reshape(T, D * C)           # (T, F)
        y = self.labels[i]
        return x, y
