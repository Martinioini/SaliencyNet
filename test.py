import torch

ckpt = torch.load('best.pt', map_location='cpu', weights_only=False)
enc = ckpt['encoder']   # lo state_dict dell'encoder

# 1. c'è un gamma? (quasi certamente no in quella run)
gamma_keys = [k for k in enc if 'gamma' in k]
print("gamma presente:", gamma_keys if gamma_keys else "NO")

# 2. che chiavi ha l'attention?
att_keys = [k for k in enc if 'att' in k]
print("chiavi attention:", att_keys)

# 3. i pesi dell'attention: quanto sono lontani dall'inizializzazione?
for k in att_keys:
    w = enc[k]
    print(f"{k:35s}  mean={w.mean().item():+.4f}  std={w.std().item():.4f}  absmax={w.abs().max().item():.4f}")