import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse, os, torch, torch.nn as nn, numpy as np
from torch.utils.data import DataLoader
from loguru import logger
from src.data.csi_dataset import CSIDataset
from src.data.csi_multiap_dataset import CSIMultiAPDataset
from src.models.who_encoder import WhoFiEncoder, WhoFiEncoderMultiAP
from src.preprocess.augment import random_augment

def collate_single(batch):
    xs, ys = zip(*batch)
    # pad T to min T across batch
    T = min(x.shape[0] for x in xs)
    xs = [x[:T] for x in xs]
    return torch.from_numpy(np.stack(xs,0)), torch.tensor(ys, dtype=torch.long)

def collate_multi(batch):
    # batch of (list_of_seqs, y)
    ys = [b[1] for b in batch]
    # find min T/D across all aps & items
    Amax = max(len(b[0]) for b in batch)
    T = min(min(s.shape[0] for s in b[0]) for b in batch)
    D = min(min(s.shape[1] for s in b[0]) for b in batch)
    Xs = []
    for a in range(Amax):
        collect = []
        for seqs,_ in batch:
            if a < len(seqs):
                s = seqs[a][:T,:D,:]
            else:
                s = np.zeros_like(seqs[0][:T,:D,:])
            collect.append(s.reshape(T, D*2))
        Xs.append(torch.from_numpy(np.stack(collect,0)))  # (N,T,F)
    return Xs, torch.tensor(ys, dtype=torch.long)

def apply_aug(batch, is_multi, p=0.8):
    if not is_multi:
        X, y = batch
        Xn = []
        for i in range(X.size(0)):
            seq = X[i].numpy().reshape(X.size(1), -1, 2)
            if np.random.rand()<p: seq = random_augment(seq)
            Xn.append(seq.reshape(X.size(1), -1))
        return torch.from_numpy(np.stack(Xn,0)), y
    else:
        Xs, y = batch
        Xs_out=[]
        for a, Xa in enumerate(Xs):
            Xn=[]
            for i in range(Xa.size(0)):
                seq = Xa[i].numpy().reshape(Xa.size(1), -1, 2)
                if np.random.rand()<p: seq = random_augment(seq)
                Xn.append(seq.reshape(Xa.size(1), -1))
            Xs_out.append(torch.from_numpy(np.stack(Xn,0)))
        return Xs_out, y

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.multi_ap:
        train_ds = CSIMultiAPDataset(args.data_root, args.train_list, min_aps=1, train_aug=True)
        val_ds   = CSIMultiAPDataset(args.data_root, args.val_list,   min_aps=1, train_aug=False)
        train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True, collate_fn=collate_multi)
        val_loader   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False, num_workers=2, collate_fn=collate_multi)
        # infer feat_dim from one AP
        sample, _ = train_ds[0]
        T,D,_ = sample[0].shape
        feat_dim = D*2
        num_ids  = len(set(train_ds.labels))
        model = WhoFiEncoderMultiAP(feat_dim, num_ids, d_model=args.d_model, nhead=args.nhead,
                                    nlayers=args.nlayers, fusion=args.fusion, max_aps=min( len(sample), args.max_aps )).to(device)
        coll_val = collate_multi
        is_multi = True
    else:
        train_ds = CSIDataset(args.data_root, args.train_list)
        val_ds   = CSIDataset(args.data_root, args.val_list)
        train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True, collate_fn=collate_single)
        val_loader   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False, num_workers=2, collate_fn=collate_single)
        feat_dim = train_ds[0][0].shape[-1]
        num_ids  = len(set(train_ds.labels))
        model = WhoFiEncoder(feat_dim, num_ids, d_model=args.d_model, nhead=args.nhead, nlayers=args.nlayers).to(device)
        coll_val = collate_single
        is_multi = False

    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            batch = apply_aug(batch, is_multi, p=args.aug_prob)
            if is_multi:
                Xs, y = batch
                Xs = [x.to(device) for x in Xs]; y = y.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits, _ = model(Xs)
                    loss = crit(logits, y)
            else:
                X, y = batch
                X, y = X.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits, _ = model(X)
                    loss = crit(logits, y)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()

        model.eval(); correct=0; total=0
        with torch.no_grad():
            for batch in val_loader:
                if is_multi:
                    Xs, y = coll_val(batch)
                    Xs = [x.to(device) for x in Xs]; y = y.to(device)
                    logits,_ = model(Xs)
                else:
                    X, y = coll_val(batch)
                    X, y = X.to(device), y.to(device)
                    logits,_ = model(X)
                pred = logits.argmax(-1)
                correct += (pred==y).sum().item()
                total += y.numel()
        acc = correct / max(1,total)
        logger.info(f"[epoch {epoch}] val@1={acc:.3f}")
        if acc>best:
            best=acc
            os.makedirs(args.out, exist_ok=True)
            torch.save({'model':model.state_dict(),'feat_dim':feat_dim,'num_ids':num_ids,
                        'multi_ap':args.multi_ap,'fusion':args.fusion,'max_aps':args.max_aps},
                       os.path.join(args.out, "who_reid_best_multiap.pth"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--train_list", default=None)
    ap.add_argument("--val_list",   default=None)
    ap.add_argument("--out", default="env/weights")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--nlayers", type=int, default=6)
    ap.add_argument("--multi_ap", action="store_true")
    ap.add_argument("--fusion", choices=["concat","attn"], default="concat")
    ap.add_argument("--max_aps", type=int, default=4)
    ap.add_argument("--aug_prob", type=float, default=0.8)
    args = ap.parse_args()
    main(args)
