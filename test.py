import argparse
import torch

# path di default coerente col nuovo schema di naming (define_model_path in training.py)
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models/best_model_50_att_skip.pt",
                    help="Checkpoint da ispezionare (default: resnet50 con attention e skip)")
args = parser.parse_args()

ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
att = ckpt['attention']   # ora l'attention è un modulo separato

print(f"checkpoint: {args.model}")
print("chiavi attention:", list(att.keys()))
print()

# IL numero che conta: gamma. Se ~0, l'attention è spenta.
if 'gamma_c3' in att:
    print(f">>> GAMMA C3 = {att['gamma_c3'].item():.6f}")
    print("    (se ~0 l'attention e' spenta; se cresciuto, e' attiva)\n")

if 'gamma_c4' in att:
    print(f">>> GAMMA C4 = {att['gamma_c4'].item():.6f}")
    print("    (se ~0 l'attention e' spenta; se cresciuto, e' attiva)\n")

# statistiche dei pesi dell'attention
for k in att:
    w = att[k]
    if w.numel() > 1:
        print(f"{k:30s}  std={w.std().item():.4f}  absmax={w.abs().max().item():.4f}")
    else:
        print(f"{k:30s}  value={w.item():+.6f}")
