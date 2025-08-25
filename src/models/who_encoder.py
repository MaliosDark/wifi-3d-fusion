import torch, torch.nn as nn, math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos * div)
        pe[:,1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(1))  # (L,1,D)

    def forward(self, x):  # x: (L,N,D)
        L = x.size(0)
        return x + self.pe[:L]

class WhoFiEncoder(nn.Module):
    """
    Transformer encoder producing an ID classifier + normalized embedding.
    Input:  (N, T, F)  where F = D*2 (amp+phase flattened per frame).
    Output: logits (N,num_ids), embedding (N,emb_dim).
    """
    def __init__(self, feat_dim: int, num_ids: int,
                 d_model=256, nhead=8, nlayers=6, dim_ff=512, dropout=0.1, emb_dim=256):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.cls = nn.Parameter(torch.zeros(1,1,d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=False)
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.pos = PositionalEncoding(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_ids)
        self.emb = nn.Linear(d_model, emb_dim)

    def forward(self, x):  # (N,T,F)
        N,T,F = x.shape
        x = self.input_proj(x)           # (N,T,D)
        x = x.transpose(0,1)             # (T,N,D)
        x = self.pos(x)
        cls = self.cls.expand(-1, N, -1) # (1,N,D)
        x = torch.cat([cls, x], dim=0)   # (T+1,N,D)
        x = self.encoder(x)              # (T+1,N,D)
        x = self.norm(x[0])              # (N,D)  # CLS
        logits = self.head(x)
        emb = self.emb(x)
        emb = nn.functional.normalize(emb, dim=-1)
        return logits, emb


class WhoFiEncoderMultiAP(nn.Module):
    """
    Multi-AP encoder with early concat fusion or late attention fusion.
    Inputs: list of tensors per AP shaped (N,T,F)
    fusion: 'concat' or 'attn'
    """
    def __init__(self, feat_dim:int, num_ids:int, d_model=256, nhead=8, nlayers=6, dim_ff=512, dropout=0.1, emb_dim=256, fusion:str='concat', max_aps:int=4):
        super().__init__()
        self.fusion = fusion
        self.max_aps = max_aps
        self.input_proj = nn.ModuleList([nn.Linear(feat_dim, d_model) for _ in range(max_aps)])
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=False)
        self.encoder = nn.ModuleList([nn.TransformerEncoder(enc_layer, nlayers) for _ in range(max_aps)])
        self.cls = nn.Parameter(torch.zeros(1,1,d_model))
        self.pos = PositionalEncoding(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model if fusion=='attn' else d_model*max_aps, num_ids)
        self.emb  = nn.Linear(d_model if fusion=='attn' else d_model*max_aps, emb_dim)

        if fusion=='attn':
            self.ap_query = nn.Parameter(torch.randn(1,1,d_model))
            self.ap_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

    def _encode_ap(self, x, ap_idx):
        # x: (N,T,F) -> (N,D)
        N,T,F = x.shape
        proj = self.input_proj[ap_idx](x)     # (N,T,D)
        z = proj.transpose(0,1)               # (T,N,D)
        z = self.pos(z)
        cls = self.cls.expand(1,N,-1)         # (1,N,D)
        z = torch.cat([cls, z], dim=0)        # (T+1,N,D)
        z = self.encoder[ap_idx](z)
        return self.norm(z[0])                # (N,D)

    def forward(self, xs: list):
        # xs: list of (N,T,F)
        A = min(len(xs), self.max_aps)
        encs = [self._encode_ap(xs[a], a) for a in range(A)]  # each (N,D)
        if self.fusion == 'concat':
            feat = torch.cat(encs + [torch.zeros_like(encs[0])]*(self.max_aps-A), dim=-1) if A<self.max_aps else torch.cat(encs, dim=-1)
        else:
            # attention over AP embeddings
            H = torch.stack(encs, dim=1)     # (N,A,D)
            q = self.ap_query.expand(H.size(0),1,-1)  # (N,1,D)
            out,_ = self.ap_attn(q, H, H)    # (N,1,D)
            feat = out.squeeze(1)            # (N,D)
        logits = self.head(feat)
        emb = nn.functional.normalize(self.emb(feat), dim=-1)
        return logits, emb
