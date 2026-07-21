import argparse
import torch

# default path consistent with define_model_path
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models/best_model_50_att_skip.pt",
                    help="checkpoint to inspect")
args = parser.parse_args()

ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
att = ckpt['attention']

print(f"checkpoint: {args.model}")
print("attention keys:", list(att.keys()))
print()

# gamma matters. if roughly 0, attention is switched off by the network.
if 'gamma_c3' in att:
    print(f">>> GAMMA C3 = {att['gamma_c3'].item():.6f}")
    print("    (if ~0 attention is off; if grown, it is active)\n")

if 'gamma_c4' in att:
    print(f">>> GAMMA C4 = {att['gamma_c4'].item():.6f}")
    print("    (if ~0 attention is off; if grown, it is active)\n")

# attention weights stats
for k in att:
    w = att[k]
    if w.numel() > 1:
        print(f"{k:30s}  std={w.std().item():.4f}  absmax={w.abs().max().item():.4f}")
    else:
        print(f"{k:30s}  value={w.item():+.6f}")
