import matplotlib
matplotlib.use('Agg')  # backend non interattivo: plt.show() bloccherebbe lo script
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import random

#Pearson correlation between pred and target, per image
def cc_metric(pred, target, eps=1e-8):

    B = pred.size(0)
    p = pred.view(B, -1).float()
    t = target.view(B, -1).float()

    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)

    num = (p * t).sum(dim=1)
    den = torch.sqrt((p * p).sum(dim=1) * (t * t).sum(dim=1) + eps)

    return (num / den)  # per-sample, mediato fuori


#Histogram intersection between the two distributions, per image
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

    return sim_per_image  # per-sample, mediato fuori


#KL divergence, takes raw logits and does its own log_softmax
def kl_metric(pred, ground_truth, eps=1e-8):

    pred = pred.float()
    ground_truth = ground_truth.float()

    B = pred.size(0)
    pred = pred.view(B, -1)
    targ = ground_truth.view(B, -1)

    log_pred = F.log_softmax(pred, dim=1)
    targ = targ / (targ.sum(dim=1, keepdim=True) + eps)

    return F.kl_div(log_pred, targ, reduction='batchmean')


#Normalized Scanpath Saliency, uses the fixation map instead of the continuous one
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

    return nss  # per-sample, mediato fuori


def evaluate_loader(image_encoder, image_decoder, image_attention, device, loader, label):
    #Metrics setup: load saved best model (caller loads the checkpoint beforehand)

    # accumulo la SOMMA per-campione di ogni metrica e divido per il numero
    # totale di campioni alla fine: cosi' l'ultimo batch (piu' piccolo con
    # drop_last=False) pesa in proporzione al numero di immagini, non uguale
    # a un batch pieno.
    sim_sum = 0.0
    cc_sum = 0.0
    nss_sum = 0.0
    kl_sum = 0.0
    n_total = 0

    #Metrics over the given loader:
    with torch.no_grad():

        for image_batch, image_map_batch, image_fix_batch in loader:

                image_batch = image_batch.to(device)
                image_map_batch = image_map_batch.to(device)
                image_fix_batch = image_fix_batch.to(device)

                c1, c2, c3, c4 = image_encoder(image_batch)
                c3_att, c4_att = image_attention(c3, c4)
                decoded_data = image_decoder(c1, c2, c3_att, c4_att)

                B = image_batch.shape[0]

                flattened_decoded_data = decoded_data.view(B, -1)
                log_decoded_data = F.softmax(flattened_decoded_data, dim=1)
                normalized_decoded_data = log_decoded_data.view_as(decoded_data)
                cc_norm_decoded_data = normalized_decoded_data / (normalized_decoded_data.amax(dim=(2, 3), keepdim=True) + 1e-8)

                sim_sum += sim_metric(normalized_decoded_data, image_map_batch).sum().item()
                cc_sum += cc_metric(cc_norm_decoded_data, image_map_batch).sum().item()
                nss_sum += nss_metric(normalized_decoded_data, image_fix_batch).sum().item()
                # KL e' batchmean: rimoltiplico per B per recuperare la somma
                kl_sum += kl_metric(decoded_data, image_map_batch).item() * B
                n_total += B

    sim = sim_sum / n_total
    cc = cc_sum / n_total
    nss = nss_sum / n_total
    kl = kl_sum / n_total

    print(f'\n{label} METRICS: \nSIM: {sim}\n CC: {cc}\n NSS: {nss}\n KL: {kl}\n')

    return sim, cc, nss, kl


def predict_normalized(image_encoder, image_decoder, image_attention, imgs):
    c1, c2, c3, c4 = image_encoder(imgs)
    c3_att, c4_att = image_attention(c3, c4)
    logits = image_decoder(c1, c2, c3_att, c4_att)
    B = logits.size(0)
    prob = F.softmax(logits.view(B, -1), dim=1).view_as(logits)
    pred = prob / (prob.amax(dim=(2, 3), keepdim=True) + 1e-8)
    return pred, logits


def compare_to_center_baseline(image_encoder, image_decoder, image_attention, device, val_loader):
    image_encoder.eval()
    image_decoder.eval()
    image_attention.eval()

    # Center prior setup
    H = W = 256
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    center = torch.exp(-((yy - H/2)**2 + (xx - W/2)**2) / (2 * (H/4)**2)).float().to(device)
    
    center_norm = (center / (center.amax() + 1e-8)).view(1, 1, H, W)
    center_logits = torch.log(center / center.sum() + 1e-8).view(1, 1, H, W)

    # somme per-campione (stessa aggregazione di evaluate_loader)
    m_cc_sum = m_nss_sum = m_kl_sum = 0.0
    c_cc_sum = c_nss_sum = c_kl_sum = 0.0
    n_total = 0

    print("Evaluating Center Baseline vs Model...")

    with torch.no_grad():
        for i, (image_batch, image_map_batch, image_fix_batch) in enumerate(val_loader):
            
            image_batch = image_batch.to(device)
            image_map_batch = image_map_batch.to(device)
            image_fix_batch = image_fix_batch.to(device)
            B = image_batch.shape[0]

            c1, c2, c3, c4 = image_encoder(image_batch)
            c3_att, c4_att = image_attention(c3, c4)
            decoded_data = image_decoder(c1, c2, c3_att, c4_att)

            flattened_decoded_data = decoded_data.view(B, -1)
            log_decoded_data = F.softmax(flattened_decoded_data, dim=1)
            normalized_decoded_data = log_decoded_data.view_as(decoded_data)
            cc_norm_decoded_data = normalized_decoded_data / (normalized_decoded_data.amax(dim=(2, 3), keepdim=True) + 1e-8)

            batch_center_norm = center_norm.expand(B, -1, -1, -1)
            batch_center_logits = center_logits.expand(B, -1, -1, -1)

            m_cc_sum += cc_metric(cc_norm_decoded_data, image_map_batch).sum().item()
            m_nss_sum += nss_metric(normalized_decoded_data, image_fix_batch).sum().item()
            m_kl_sum += kl_metric(decoded_data, image_map_batch).item() * B

            c_cc_sum += cc_metric(batch_center_norm, image_map_batch).sum().item()
            c_nss_sum += nss_metric(batch_center_norm, image_fix_batch).sum().item()
            c_kl_sum += kl_metric(batch_center_logits, image_map_batch).item() * B

            n_total += B

            # Plot samples during the first batch only
            if i == 0:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                
                n = min(4, image_batch.shape[0])
                fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
                
                for j in range(n):
                    img_show = (image_batch[j].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0)
                    
                    axes[j, 0].imshow(img_show)
                    axes[j, 0].set_title("Input")
                    axes[j, 0].axis('off')
                    
                    axes[j, 1].imshow(image_map_batch[j, 0].cpu(), cmap='hot')
                    axes[j, 1].set_title("Ground Truth")
                    axes[j, 1].axis('off')
                    
                    axes[j, 2].imshow(normalized_decoded_data[j, 0].cpu(), cmap='hot')
                    axes[j, 2].set_title("Prediction")
                    axes[j, 2].axis('off')
                    
                    axes[j, 3].imshow(image_fix_batch[j, 0].cpu(), cmap='hot')
                    axes[j, 3].set_title("Fixation")
                    axes[j, 3].axis('off')
                    
                Path('plots').mkdir(parents=True, exist_ok=True)
                plt.tight_layout()
                plt.savefig('plots/center_baseline.png')
                plt.close()

    m_cc = m_cc_sum / n_total
    m_nss = m_nss_sum / n_total
    m_kl = m_kl_sum / n_total

    c_cc = c_cc_sum / n_total
    c_nss = c_nss_sum / n_total
    c_kl = c_kl_sum / n_total

    print("\nBaseline Comparison Results:")
    print(f"KL  -> Model: {m_kl:.4f} | Center: {c_kl:.4f} | Difference: {m_kl - c_kl:+.4f}")
    print(f"CC  -> Model: {m_cc:.4f} | Center: {c_cc:.4f} | Difference: {m_cc - c_cc:+.4f}")
    print(f"NSS -> Model: {m_nss:.4f} | Center: {c_nss:.4f} | Difference: {m_nss - c_nss:+.4f}\n")

    return m_cc, m_nss, m_kl, c_cc, c_nss, c_kl
