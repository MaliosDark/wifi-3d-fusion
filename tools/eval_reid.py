import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse, torch, numpy as np
from torch.utils.data import DataLoader
from src.data.csi_dataset import CSIDataset
from src.models.who_encoder import WhoFiEncoder

def embeddings(loader, model, device):
    feats, labels = [], []
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            _, emb = model(x)
            feats.append(emb.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(feats,0), np.concatenate(labels,0)

def cmc_map(query_f, query_y, gal_f, gal_y):
    # cosine similarity
    q = query_f / (np.linalg.norm(query_f,axis=1,keepdims=True)+1e-9)
    g = gal_f   / (np.linalg.norm(gal_f,axis=1,keepdims=True)+1e-9)
    sim = q @ g.T
    idx = np.argsort(-sim, axis=1)
    rank1 = (gal_y[idx[:,0]] == query_y).mean()

    APs = []
    for i in range(q.shape[0]):
        order = idx[i]
        matches = (gal_y[order] == query_y[i]).astype(np.int32)
        if matches.sum()==0: continue
        cum = np.cumsum(matches)
        prec = cum / (np.arange(len(matches))+1)
        APs.append(((prec * matches).sum() / matches.sum()))
    mAP = float(np.mean(APs)) if APs else 0.0
    return float(rank1), float(mAP)

def main(a):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds_q = CSIDataset(a.data_root, a.query_list)
    ds_g = CSIDataset(a.data_root, a.gallery_list)

    feat_dim = ds_q[0][0].shape[-1]
    ckpt = torch.load(a.checkpoint, map_location='cpu')
    model = WhoFiEncoder(feat_dim, ckpt.get('num_ids', 100))
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(device)

    ql = DataLoader(ds_q, batch_size=a.bs, shuffle=False)
    gl = DataLoader(ds_g, batch_size=a.bs, shuffle=False)
    qf, qy = embeddings(ql, model, device)
    gf, gy = embeddings(gl, model, device)
    r1, mp = cmc_map(qf, qy, gf, gy)
    print(f"Rank-1: {r1:.4f} | mAP: {mp:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--query_list", required=True)
    ap.add_argument("--gallery_list", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()
    main(a)
