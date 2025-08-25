import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse, os, torch, torch.nn as nn
from torch.utils.data import DataLoader
from src.data.csi_dataset import CSIDataset
from src.models.who_encoder import WhoFiEncoder

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = CSIDataset(args.data_root, args.train_list)
    val_ds   = CSIDataset(args.data_root, args.val_list)
    feat_dim = train_ds[0][0].shape[-1]
    num_ids  = len(set(train_ds.labels))
    model = WhoFiEncoder(feat_dim, num_ids, d_model=args.d_model, nhead=args.nhead, nlayers=args.nlayers).to(device)

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=2)

    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits, _ = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()

        # validation (top-1)
        model.eval(); correct=0; total=0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                logits,_ = model(x)
                pred = logits.argmax(-1)
                correct += (pred==y).sum().item()
                total += y.numel()
        acc = correct / max(1,total)
        print(f"[epoch {epoch}] val@1={acc:.3f}")
        if acc>best:
            best=acc
            os.makedirs(args.out, exist_ok=True)
            torch.save({'model':model.state_dict(),'feat_dim':feat_dim,'num_ids':num_ids},
                       os.path.join(args.out, "who_reid_best.pth"))

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
    args = ap.parse_args()
    main(args)
