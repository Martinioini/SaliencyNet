import csv
import matplotlib
matplotlib.use('Agg')  # non-interactive backend: plt.show() would block the script when running headless
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from models import *

CURVE_LABELS = {
    'resnet18_noskip_noatt': 'Base',
    'resnet18_skip_noatt': '+skip',
    'resnet18_skip_att': '+attention',
    'resnet50_skip_att': '+ResNet50',
}


# Training function. When encoder_frozen is True, we place the encoder and attention in eval mode
# to lock their Batch Normalization statistics and disable dropout, preserving the pretrained features.
def train_epoch(encoder, decoder, attention, device, dataloader, loss_fn, optimizer, encoder_frozen=False):
    if encoder_frozen:
        encoder.eval()
        attention.eval()
    else :
        encoder.train()
        attention.train()
    decoder.train()
    losses = []

    for image_batch, image_map_batch, image_fix_batch in dataloader:

        image_batch = image_batch.to(device)
        image_map_batch = image_map_batch.to(device)

        c1, c2, c3, c4 = encoder(image_batch)
        c3_att, c4_att = attention(c3, c4)
        decoded_data = decoder(c1, c2, c3_att, c4_att)

        loss = loss_fn(decoded_data, image_map_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu().numpy())
    losses = np.mean(losses)
    return losses

#testing function, gradients are not kept and both models are in eval mode
def test_epoch(encoder, decoder, attention, device, dataloader, loss_fn):
    encoder.eval()
    decoder.eval()
    attention.eval()

    val_losses = []

    with torch.no_grad():

        for image_batch, image_map_batch, image_fix_batch in dataloader:
            image_batch = image_batch.to(device)
            image_map_batch = image_map_batch.to(device)

            c1, c2, c3, c4 = encoder(image_batch)
            c3_att, c4_att = attention(c3, c4)
            decoded_data = decoder(c1, c2, c3_att, c4_att)

            loss = loss_fn(decoded_data, image_map_batch)
            val_losses.append(loss.item())

    return np.mean(val_losses)

# Phase 1: Train the decoder alone with a frozen encoder. This warms up the decoder weights
# without exposing the pretrained backbone to the noisy gradients of a randomly initialized decoder.
def run_phase1(image_encoder, image_decoder, image_attention, device, train_loader, val_loader, loss_fn, optim1,
               train_loss_history, val_loss_history, use_attention, use_skip, model_name="model", num_epochs=5, patience=3):
    bad = 0
    best_val_error = float('inf')

    models_dir = Path('models')
    models_dir.mkdir(parents=True, exist_ok=True)
    Path('plots').mkdir(parents=True, exist_ok=True)

    for p in image_encoder.parameters():
        p.requires_grad = False

    for epoch in range(num_epochs):
        print('EPOCH %d/%d' % (epoch + 1, num_epochs))
        # Training (use the testing function)
        train_loss = train_epoch(
            encoder=image_encoder,
            decoder=image_decoder,
            attention=image_attention,
            device=device,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optim1,
            encoder_frozen=True)
        print(f'TRAIN - EPOCH {epoch+1}/{num_epochs} - loss: {train_loss}')

        # Validation (use the testing function)
        val_loss = test_epoch(
            encoder=image_encoder,
            decoder=image_decoder,
            attention=image_attention,
            device=device,
            dataloader=val_loader,
            loss_fn=loss_fn)

        # Print Validation curves
        print(f'VALIDATION - EPOCH {epoch+1}/{num_epochs} - loss: {val_loss}\n')
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)


        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_history, label='Train Loss (Phase 1)', color='lightblue')
        plt.plot(val_loss_history, label='Val Loss (Phase 1)', color='navajowhite')
        plt.title(f'Learning Curves - {model_name}')
        plt.xlabel('Epoch')
        plt.ylabel('KL Divergence Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'plots/{model_name}_curves.png') # Salva col nome del modello
        plt.close()

        #early stopping and saving of the best model
        if best_val_error > val_loss:
            best_val_error = val_loss
            bad = 0

            models_path = define_model_path(image_encoder, use_skip, use_attention)
            torch.save({"encoder" : image_encoder.state_dict(), "decoder" : image_decoder.state_dict(), "attention" : image_attention.state_dict()}, models_path)
        else:
            bad += 1
            if bad == patience:
                break

# Phase 2: Fine-tune the entire network jointly. We unfreeze the backbone and train
# all parameters using differential learning rates since different layers need to move different amounts.
def run_phase2(image_encoder, image_decoder, image_attention, device, train_loader, val_loader, loss_fn, optim, scheduler,
               train_loss_history, val_loss_history, use_attention, use_skip, model_name="model", num_epochs=50, patience=3):
    
    phase1_len = len(train_loss_history)

    bad = 0
    best_val_error = float('inf')
    models_dir = Path('models')
    models_dir.mkdir(parents=True, exist_ok=True)
    Path('plots').mkdir(parents=True, exist_ok=True)

    image_encoder.train()
    image_decoder.train()
    image_attention.train()

    #unfreeze the encoder and train on the whole net
    for p in image_encoder.parameters():
        p.requires_grad = True

    for epoch in range(num_epochs):
        print('EPOCH %d/%d' % (epoch + 1, num_epochs))
        # Training (use the training function)
        train_loss = train_epoch(
            encoder=image_encoder,
            decoder=image_decoder,
            attention=image_attention,
            device=device,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optim)
        print(f'TRAIN - EPOCH {epoch+1}/{num_epochs} - loss: {train_loss}')

        # Validation (use the testing function)
        val_loss = test_epoch(
            encoder=image_encoder,
            decoder=image_decoder,
            attention=image_attention,
            device=device,
            dataloader=val_loader,
            loss_fn=loss_fn)

        scheduler.step(val_loss)

        # Print Validation curves
        print(f'VALIDATION - EPOCH {epoch+1}/{num_epochs} - loss: {val_loss}\n')
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        plt.figure(figsize=(10, 5))
        
        if phase1_len > 0:
            plt.plot(range(phase1_len), train_loss_history[:phase1_len], label='Train Loss (Phase 1)', color='lightblue')
            plt.plot(range(phase1_len), val_loss_history[:phase1_len], label='Val Loss (Phase 1)', color='navajowhite')

        start_idx = max(0, phase1_len - 1)
        plt.plot(range(start_idx, len(train_loss_history)), train_loss_history[start_idx:], label='Train Loss (Phase 2)', color='blue')
        plt.plot(range(start_idx, len(val_loss_history)), val_loss_history[start_idx:], label='Val Loss (Phase 2)', color='orange')
        plt.axvline(x=start_idx, color='grey', linestyle='--', label='Start Phase 2')
        plt.title(f'Learning Curves - {model_name}')
        plt.xlabel('Epoch')
        plt.ylabel('KL Divergence Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'plots/{model_name}_curves.png')
        plt.close()

        #early stopping and saving of the best model
        if best_val_error > val_loss:
            best_val_error = val_loss
            bad = 0
            path = define_model_path(image_encoder, use_skip, use_attention)
            torch.save({"encoder" : image_encoder.state_dict(), "decoder" : image_decoder.state_dict(), "attention" : image_attention.state_dict()}, path)
        else:
            bad += 1
            if bad == patience:
                break

def define_model_path(image_encoder, use_skip, use_attention):
    path = Path('models')
    attention = 'att' if use_attention else 'noatt'
    skip = 'skip' if use_skip else 'noskip'
    
    if isinstance(image_encoder, encoder50):
        return path / f'best_model_50_{attention}_{skip}.pt'
    elif isinstance(image_encoder, encoder18):
        return path / f'best_model_18_{attention}_{skip}.pt'

# Overlays the validation curve of every completed run onto one figure for the paper.
def plot_all_curves():
    csv_paths = sorted(Path('plots').glob('*_curves.csv'))

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    plt.rcParams.update({'font.size': 8})

    drawn_boundaries = set()

    for csv_path in csv_paths:
        tag = csv_path.stem[:-len('_curves')]

        epochs, val_losses, phases = [], [], []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row['epoch']))
                val_losses.append(float(row['val_loss']))
                phases.append(int(row['phase']))

        if not epochs:
            continue

        label = CURVE_LABELS.get(tag, tag)
        line, = ax.plot(epochs, val_losses, label=label, linewidth=1)

        best_idx = int(np.argmin(val_losses))
        ax.plot(epochs[best_idx], val_losses[best_idx], 'o', color=line.get_color(), markersize=3)

        phase1_epochs = [e for e, p in zip(epochs, phases) if p == 1]
        if phase1_epochs and phase1_epochs[-1] not in drawn_boundaries:
            ax.axvline(x=phase1_epochs[-1], color='grey', linestyle='--', linewidth=0.8)
            drawn_boundaries.add(phase1_epochs[-1])

    ax.set_xlabel('Epoch')
    ax.set_ylabel('KL Divergence Loss')
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig('plots/fig_curves.pdf')
    plt.close(fig)