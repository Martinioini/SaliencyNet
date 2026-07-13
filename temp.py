import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import prepare_paths, ImageDataset
from models import *
from losses import SaliencyKLLoss
from training import *
from metrics import evaluate_loader, compare_to_center_baseline
import argparse

# --- config (stessi valori usati nel notebook) ---
ROOT = './datasets/salicon'
BATCH_SIZE = 16
NUM_WORKERS = 8
PIN_MEMORY = True  # ATTENZIONE: IN LOCALE SU GPU METTI pin_memory = True

PHASE1_LR = 1e-3
PHASE1_WEIGHT_DECAY = 1e-5
PHASE1_EPOCHS = 6
PHASE1_PATIENCE = 3

PHASE2_ENCODER_LR = 1e-5
PHASE2_DECODER_LR = 1e-4
PHASE2_ATTENTION_LR = 1e-3
PHASE2_WEIGHT_DECAY = 1e-5
PHASE2_EPOCHS = 50
PHASE2_PATIENCE = 5
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3
SCHEDULER_MIN_LR = 1e-7


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["resnet18", "resnet50"], default="resnet18", help="Choose the backbone: resnet18 or resnet50")    
    parser.add_argument("run_type", choices=["train_18", "train_50", "test_18", "test_50"], help="Choose the action: train_18, train_50, test_18, or test_50")
    parser.add_argument("--no_skip", action="store_true", help="Disable skip connections")
    parser.add_argument("--no_attention", action="store_true", help="Disable attention mechanism")
    args = parser.parse_args()


    use_skip = not args.no_skip
    use_attention = not args.no_attention  

    if args.run_type == "train_18":
        backbone_type = "resnet18"
        do_train = True
    elif args.run_type == "train_50":
        backbone_type = "resnet50"
        do_train = True
    elif args.run_type == "test_18":
        backbone_type = "resnet18"
        do_train = False
    elif args.run_type == "test_50":
        backbone_type = "resnet50"
        do_train = False
    
    print(f"Using backbone: {backbone_type}, Skip connections: {use_skip}, Attention: {use_attention}")
    
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    root = Path(ROOT)
    train_dir = root / "images/images"
    fix_dir = root / "fixations"
    map_dir = root / "maps"

    paths = prepare_paths(train_dir, map_dir, fix_dir)
    (image_paths_train, map_paths_train, fix_paths_train) = paths["train"]
    (image_paths_val, map_paths_val, fix_paths_val) = paths["val"]
    (image_paths_test, map_paths_test, fix_paths_test) = paths["test"]

    train_ds = ImageDataset(image_paths_train, map_paths_train, fix_paths_train, train=True)
    val_ds = ImageDataset(image_paths_val, map_paths_val, fix_paths_val, train=False)
    test_ds = ImageDataset(image_paths_test, map_paths_test, fix_paths_test, train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    if backbone_type == "resnet50":
        image_encoder = encoder50()
        image_decoder = decoder50(use_skip=use_skip)
        image_attention = attention50(use_attention=use_attention)

    elif backbone_type == "resnet18":
        image_encoder = encoder18()
        image_decoder = decoder18(use_skip=use_skip)
        image_attention = attention18(use_attention=use_attention)

    image_encoder.to(device)
    image_decoder.to(device)
    image_attention.to(device)

    loss_fn = SaliencyKLLoss()

    if do_train:
        train_loss_history = []
        val_loss_history = []

        # --- Phase 1: encoder frozen, decoder only ---
        params_to_optimize1 = [
            {'params': image_decoder.parameters(), 'lr': PHASE1_LR}
        ]
        optim1 = torch.optim.Adam(params_to_optimize1, weight_decay=PHASE1_WEIGHT_DECAY)

        run_phase1(
            image_encoder, image_decoder, image_attention, device, train_loader, val_loader, loss_fn, optim1,
            train_loss_history, val_loss_history, use_attention=use_attention, use_skip=use_skip,
            model_name=backbone_type, num_epochs=PHASE1_EPOCHS, patience=PHASE1_PATIENCE)

        # carica il best della fase 1 prima di sbloccare l'encoder (l'ultima epoca puo' essere peggiore)
        phase1_ckpt_path = define_model_path(image_encoder, use_skip, use_attention)
        ckpt1 = torch.load(phase1_ckpt_path, map_location=device, weights_only=False)
        image_encoder.load_state_dict(ckpt1['encoder'])
        image_decoder.load_state_dict(ckpt1['decoder'])
        image_attention.load_state_dict(ckpt1['attention'])

        # --- Phase 2: whole network, differential LR ---
        params_to_optimize = [
            {'params': image_encoder.parameters(), 'lr': PHASE2_ENCODER_LR},
            {'params': image_decoder.parameters(), 'lr': PHASE2_DECODER_LR},
            {'params': image_attention.parameters(), 'lr': PHASE2_ATTENTION_LR},
        ]
        optim = torch.optim.Adam(params_to_optimize, weight_decay=PHASE2_WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR)
        image_encoder.to(device)
        image_decoder.to(device)

        run_phase2(
            image_encoder, image_decoder, image_attention, device, train_loader, val_loader, loss_fn, optim, scheduler,
            train_loss_history, val_loss_history, use_attention=use_attention, use_skip=use_skip,
            model_name=backbone_type, num_epochs=PHASE2_EPOCHS, patience=PHASE2_PATIENCE)

    # --- Metrics on best.pt ---
    ckpt_path = define_model_path(image_encoder, use_skip, use_attention)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    image_encoder.load_state_dict(ckpt['encoder'])
    image_decoder.load_state_dict(ckpt['decoder'])
    image_attention.load_state_dict(ckpt['attention'])
    image_encoder.eval()
    image_decoder.eval()
    image_attention.eval()

    evaluate_loader(image_encoder, image_decoder, image_attention, device, val_loader, "VALIDATION")
    evaluate_loader(image_encoder, image_decoder, image_attention, device, test_loader, "TEST")

    compare_to_center_baseline(image_encoder, image_decoder, image_attention, device, val_loader)


if __name__ == "__main__":
    main()

