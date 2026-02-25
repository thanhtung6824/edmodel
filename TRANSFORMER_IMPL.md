# SFP Transformer Implementation Guide

Two new files to create. Everything else (`dataset.py`, `losses.py`, `sfp_labels.py`) stays unchanged.

---

## File 1: `src/models/transformer_model.py`

### Class: `SFPTransformer(nn.Module)`

**Constructor args:**
```
n_features=14, d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.1
```

**Layers to create (in order):**

| Layer | What | Shape transform |
|-------|------|-----------------|
| `input_proj` | `nn.Linear(n_features, d_model)` | `(B, 30, 14) -> (B, 30, 64)` |
| `pos_embed` | `nn.Parameter(torch.randn(1, 30, d_model) * 0.02)` | Learnable positional encoding, added to projected input |
| `dropout` | `nn.Dropout(dropout)` | Applied after input_proj + pos_embed |
| `encoder` | `nn.TransformerEncoder(...)` | See below |
| `norm` | `nn.LayerNorm(d_model)` | Applied to pooled output |
| `tp_head` | `nn.Sequential(Linear(64,32), ReLU, Dropout(0.1), Linear(32,1), Softplus)` | `(B, 64) -> (B, 1)` |
| `sl_head` | Same structure as tp_head | `(B, 64) -> (B, 1)` |

**Building the encoder:**
```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=dim_ff,
    dropout=dropout,
    batch_first=True,        # IMPORTANT: our data is (batch, seq, feat)
    norm_first=True,         # Pre-norm (more stable training)
)
self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
```

**`forward(self, x)` logic:**
```
1. x = self.input_proj(x)           # (B,30,14) -> (B,30,64)
2. x = self.dropout(x + self.pos_embed)  # add position info
3. x = self.encoder(x)              # self-attention, (B,30,64) -> (B,30,64)
4. x = x[:, -1, :]                  # take LAST bar only (the SFP candle), -> (B,64)
5. x = self.norm(x)
6. tp = self.tp_head(x).squeeze(-1) # (B,)
7. sl = self.sl_head(x).squeeze(-1) # (B,)
8. return tp, sl
```

**Output signature** is identical to `SFPRegModel`: returns `(tp, sl)` tensors of shape `(batch,)`.

**Parameter count check:** print `sum(p.numel() for p in model.parameters())` -- expect ~40-60K.

---

## File 2: `src/train_transformer.py`

This is a copy of `src/train_reg.py` with these changes:

### Changes from `train_reg.py`:

1. **Remove** `SFPRegModel` class (lines 18-58)
2. **Remove** `RegLoss` class (lines 62-66)
3. **Add import:**
   ```python
   from src.models.transformer_model import SFPTransformer
   ```
4. **Reuse `RegLoss` from inline** (or just copy it -- it's 4 lines, your call)

5. **In `train()` function, change model creation (line 217):**
   ```python
   # OLD:
   model = SFPRegModel(n_features=n_features).to(device)
   # NEW:
   model = SFPTransformer(n_features=n_features).to(device)
   ```

6. **Change learning rate (line 219):**
   ```python
   # OLD:
   optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
   # NEW:
   optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
   ```
   Transformers typically want higher LR + more regularization. Start here, tune later.

7. **Add warmup scheduler** (replace CosineAnnealing):
   ```python
   scheduler = torch.optim.lr_scheduler.OneCycleLR(
       optimizer, max_lr=1e-3, epochs=100,
       steps_per_epoch=len(train_loader),
   )
   ```
   Move `scheduler.step()` into the **training batch loop** (after `optimizer.step()`), not after the epoch.

8. **Change save path:**
   ```python
   torch.save(model.state_dict(), "best_model_transformer.pth")
   # and the load at the end:
   model.load_state_dict(torch.load("best_model_transformer.pth", weights_only=True))
   ```

9. **Print param count after model creation:**
   ```python
   n_params = sum(p.numel() for p in model.parameters())
   print(f"SFPTransformer: {n_params:,} parameters")
   ```

### Everything else stays the same:
- `load_data_set()` -- no changes
- `evaluate()` -- no changes (model returns `(tp, sl)`, same as before)
- Training loop structure, early stopping, ratio analysis -- all the same

---

## How to run

```bash
python -m src.train_transformer
```

---

## What to check after first run

1. **Prediction variance**: are predictions different across samples?
   ```
   # Add after evaluate() in the final eval block:
   print(f"TP pred std: {tp_preds.std():.4f}, SL pred std: {sl_preds.std():.4f}")
   ```
   If both are ~0, the model is still predicting the mean (same problem as LSTM).

2. **Ratio threshold**: at ratio > 1.5, does precision beat 22% (the base rate)?

3. **Training loss**: should decrease steadily. If it explodes, lower the LR to `5e-4`.

---

## Gotchas

- `batch_first=True` on `TransformerEncoderLayer` -- without this, PyTorch expects `(seq, batch, feat)` and shapes will silently be wrong
- `norm_first=True` -- Pre-LN transformer is more stable. Without it, training can diverge with small models
- `Softplus` on heads ensures positive TP/SL output (same as LSTM model)
- The `pos_embed` shape is `(1, 30, d_model)` -- the `1` broadcasts across the batch dimension
- `OneCycleLR.step()` is called **per batch**, not per epoch (unlike CosineAnnealingLR)
