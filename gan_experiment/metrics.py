import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path


# pearson correlation. internally mean-shifts, so it evaluates linear correlation regardless of absolute scaling.
def cc_metric(pred, target, eps=1e-8):

    B = pred.size(0)
    p = pred.view(B, -1).float()
    t = target.view(B, -1).float()

    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)

    num = (p * t).sum(dim=1)
    den = torch.sqrt((p * p).sum(dim=1) * (t * t).sum(dim=1) + eps)

    return (num / den)  # per-sample, averaged outside


# histogram intersection. normalizes both maps to sum to 1 internally to compare them as proper distributions.
def sim_metric(pred, ground_truth, eps=1e-8):

    pred = pred.float()
    ground_truth = ground_truth.float()

    B = pred.size(0)
    pred_flatten = pred.view(B, -1)
    ground_truth_flatten = ground_truth.view(B, -1)

    pred_sum = pred_flatten.sum(dim=1, keepdim=True) + eps
    gt_sum = ground_truth_flatten.sum(dim=1, keepdim=True) + eps

    pred_normalized = pred_flatten / pred_sum
    ground_normalized = ground_truth_flatten / gt_sum

    minimum = torch.minimum(pred_normalized, ground_normalized)
    sim_per_image = minimum.sum(dim=1)

    return sim_per_image.mean()


# kl divergence applies its own log_softmax to raw logits
def kl_metric(pred, ground_truth, eps=1e-8):

    pred = pred.float()
    ground_truth = ground_truth.float()

    B = pred.size(0)
    pred = pred.view(B, -1)
    targ = ground_truth.view(B, -1)

    log_pred = F.log_softmax(pred, dim=1)
    targ = targ / (targ.sum(dim=1, keepdim=True) + eps)

    return F.kl_div(log_pred, targ, reduction='batchmean')


# nss is the only metric evaluated against the binary fixation map instead of the continuous density map
def nss_metric(pred, fixation_map, eps=1e-8):

    pred = pred.float()
    fixation_map = fixation_map.float()

    B = pred.size(0)
    pred_flatten = pred.view(B, -1)
    fix_flatten = fixation_map.view(B, -1)

    mean = pred_flatten.mean(dim=1, keepdim=True)
    std = pred_flatten.std(dim=1, unbiased=False, keepdim=True)
    pred_flatten = (pred_flatten - mean) / (std + eps)

    pred_flatten *= fix_flatten

    nss = pred_flatten.sum(dim=1) / (fix_flatten.sum(dim=1) + eps)

    return nss.mean()


def evaluate_loader(image_encoder, image_decoder, device, loader, label):

    sim_metrics = []
    cc_metrics = []
    nss_metrics = []
    kl_metrics = []

    # metrics over the given loader:
    with torch.no_grad():

        for image_batch, image_map_batch, image_fix_batch in loader:

            image_batch = image_batch.to(device)
            image_map_batch = image_map_batch.to(device)
            image_fix_batch = image_fix_batch.to(device)

            c1, c2, c3, c4 = image_encoder(image_batch)
            decoded_data = image_decoder(c1, c2, c3, c4)

            B = image_batch.shape[0]
            # apply softmax over the flattened map because network outputs raw logits.
            # SIM/CC/NSS renormalize internally (sum/z-score), so passing softmax output is fine.
            # KL takes raw logits directly.
            prob_decoded_data = F.softmax(decoded_data.view(B, -1), dim=1).view_as(decoded_data)

            sim_metrics.append(sim_metric(prob_decoded_data, image_map_batch))
            cc_metrics.append(cc_metric(prob_decoded_data, image_map_batch).mean())
            nss_metrics.append(nss_metric(prob_decoded_data, image_fix_batch))
            kl_metrics.append(kl_metric(decoded_data, image_map_batch))

    print(f'{label} METRICS: \nSIM: {torch.stack(sim_metrics).mean()}\n CC: {torch.stack(cc_metrics).mean()}\n NSS: {torch.stack(nss_metrics).mean()}\n KL: {torch.stack(kl_metrics).mean()}')

    return sim_metrics, cc_metrics, nss_metrics, kl_metrics


def predict_normalized(image_encoder, image_decoder, imgs):
    c1, c2, c3, c4 = image_encoder(imgs)
    logits = image_decoder(c1, c2, c3, c4)
    B = logits.size(0)
    prob = F.softmax(logits.view(B, -1), dim=1).view_as(logits)
    pred = prob / (prob.amax(dim=(2, 3), keepdim=True) + 1e-8)
    return pred, logits


def compare_to_center_baseline(image_encoder, image_decoder, device, val_loader):
    image_encoder.eval()
    image_decoder.eval()

    def predict(imgs):
        return predict_normalized(image_encoder, image_decoder, imgs)

    # center prior (photographers usually center their subjects). if we don't beat this, the model learned nothing.
    H = W = 256
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    center = torch.exp(-((yy - H/2)**2 + (xx - W/2)**2) / (2 * (H/4)**2)).float()
    center_dist = (center / center.sum()).to(device)
    log_center = torch.log(center_dist + 1e-8).view(1, -1)
    center_norm = (center / center.max()).to(device)

    print(">>> STEP 1: predictions vs GT (4 samples)")
    with torch.no_grad():
        imgs, sals, fix = next(iter(val_loader))
        imgs = imgs.to(device)
        sals = sals.to(device)
        fix = fix.to(device)
        pred, _ = predict(imgs)

    # standard imagenet reverse-normalization to visualize the raw images correctly in matplotlib
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    n = min(4, imgs.size(0))
    fig, ax = plt.subplots(n, 4, figsize=(12, 3 * n))
    for i in range(n):
        img_show = (imgs[i].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0)
        ax[i, 0].imshow(img_show); ax[i, 0].set_title("Input"); ax[i, 0].axis('off')
        ax[i, 1].imshow(sals[i, 0].cpu(), cmap='hot'); ax[i, 1].set_title("GT"); ax[i, 1].axis('off')
        ax[i, 2].imshow(pred[i, 0].cpu(), cmap='hot'); ax[i, 2].set_title("Pred"); ax[i, 2].axis('off')
        ax[i, 3].imshow(fix[i, 0].cpu(), cmap='hot'); ax[i, 3].set_title("Fixation"); ax[i, 3].axis('off')
    saved_plots = Path.cwd() / "plots"
    saved_plots.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(saved_plots / 'center_baseline.png'); plt.close()

    print("\n>>> STEP 2+3: metrics on complete val set")

    kl_model_all, cc_model_all, nss_model_all = [], [], []
    kl_center_all, cc_center_all, nss_center_all = [], [], []

    with torch.no_grad():
        for imgs, sals, fix in val_loader:
            imgs = imgs.to(device)
            sals = sals.to(device)
            fix = fix.to(device)
            B = imgs.size(0)

            pred, logits = predict(imgs)

            pred_flat = logits.view(B, -1)
            targ_flat = sals.view(B, -1)
            log_pred = F.log_softmax(pred_flat, dim=1)
            targ_norm = targ_flat / (targ_flat.sum(dim=1, keepdim=True) + 1e-8)
            kl_m = F.kl_div(log_pred, targ_norm, reduction='batchmean')
            kl_model_all.append(kl_m.item())

            sals_for_cc = sals / (sals.amax(dim=(2, 3), keepdim=True) + 1e-8)
            cc_m = cc_metric(pred, sals_for_cc)
            cc_model_all.append(cc_m.cpu().numpy())

            nss_m = nss_metric(pred, fix)
            nss_model_all.append(nss_m.cpu().numpy())

            # expand the static center prior to match batch size so we can pass it through the same batched metrics
            log_center_b = log_center.expand(B, -1)
            kl_c = F.kl_div(log_center_b, targ_norm, reduction='batchmean')
            kl_center_all.append(kl_c.item())

            center_pred = center_norm.view(1, 1, H, W).expand(B, 1, H, W)
            cc_c = cc_metric(center_pred, sals_for_cc)
            cc_center_all.append(cc_c.cpu().numpy())

            nss_c = nss_metric(center_pred, fix)
            nss_center_all.append(nss_c.cpu().numpy())

    kl_model  = np.mean(kl_model_all)
    kl_center = np.mean(kl_center_all)
    cc_model  = np.concatenate(cc_model_all).mean()
    cc_center = np.concatenate(cc_center_all).mean()
    nss_model = np.mean(nss_model_all)
    nss_center = np.mean(nss_center_all)

    print(f"\n  KL  model       : {kl_model:.4f}")
    print(f"  KL  center prior  : {kl_center:.4f}")
    print(f"  -> model beats center prior by {kl_center - kl_model:+.4f} (higher = better)")
    print()
    print(f"  CC  model       : {cc_model:.4f}")
    print(f"  CC  center prior  : {cc_center:.4f}")
    print(f"  -> model beats center prior by {cc_model - cc_center:+.4f} (higher = better)")
    print()
    print(f"  NSS model       : {nss_model:.4f}")
    print(f"  NSS center prior  : {nss_center:.4f}")
    print(f"  -> model beats center prior by {nss_model - nss_center:+.4f} (higher = better)")
    print()
    print("Quick interpretation:")
    print("  - if KL_model is VERY close to KL_center -> model learned little beyond central bias")
    print("  - if CC_model > 0.7 you are in decent territory for ResNet18 baseline")
    print("  - if CC_model ~ CC_center -> model is not learning")
    print("  - high NSS indicates peak activations fall exactly on real human fixations")

    return {
        "kl_model": kl_model, "kl_center": kl_center,
        "cc_model": cc_model, "cc_center": cc_center,
        "nss_model": nss_model, "nss_center": nss_center,
    }
