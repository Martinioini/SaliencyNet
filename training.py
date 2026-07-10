import matplotlib
matplotlib.use('Agg')  # backend non interattivo: plt.show() bloccherebbe lo script
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

# Training function, if phase1 the encoder is frozen
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
        c4_att = attention(c4)
        decoded_data = decoder(c1, c2, c3, c4_att)

        loss = loss_fn(decoded_data, image_map_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu().numpy())
    losses = np.mean(losses)
    return losses

#testing function, gradients are not kept and both models are in evak mode
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
            c4_att = attention(c4)
            decoded_data = decoder(c1, c2, c3, c4_att)

            loss = loss_fn(decoded_data, image_map_batch)
            val_losses.append(loss.item())

    return np.mean(val_losses)

#Phase 1: The encder is frozen, only a few epochs of training are done
def run_phase1(image_encoder, image_decoder, image_attention, device, train_loader, val_loader, loss_fn, optim1,
               train_loss_history, val_loss_history, num_epochs=5, patience=3):
    bad = 0
    best_val_error = float('inf')

    models_dir = Path('models')
    models_dir.mkdir(parents=True, exist_ok=True)
    Path('plots').mkdir(parents=True, exist_ok=True)

    for p in image_encoder.parameters():
        p.requires_grad = False

    for epoch in range(num_epochs):
        print('EPOCH %d/%d' % (epoch + 1, num_epochs))
        ### Training
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

        ### Validation
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
        plt.plot(train_loss_history, label='Train Loss', color='blue')
        plt.plot(val_loss_history, label='Validation Loss', color='orange')
        plt.title('Learning Curves')
        plt.xlabel('Epoch')
        plt.ylabel('KL Divergence Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/phase1_curves.png')
        plt.close()

        #early stopping and saving of the best model
        if best_val_error > val_loss:
            best_val_error = val_loss
            bad = 0
            models_path = models_dir / 'phase1.pt'
            torch.save({"encoder" : image_encoder.state_dict(), "decoder" : image_decoder.state_dict(), "attention" : image_attention.state_dict()}, models_path)
        else:
            bad += 1
            if bad == patience:
                break

#Phase 2: The whole network is used
def run_phase2(image_encoder, image_decoder, image_attention, device, train_loader, val_loader, loss_fn, optim, scheduler,
               train_loss_history, val_loss_history, num_epochs=50, patience=3):
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
        ### Training (use the training function)
        train_loss = train_epoch(
            encoder=image_encoder,
            decoder=image_decoder,
            attention=image_attention,
            device=device,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optim)
        print(f'TRAIN - EPOCH {epoch+1}/{num_epochs} - loss: {train_loss}')

        ### Validation  (use the testing function)
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
        plt.plot(train_loss_history, label='Train Loss', color='blue')
        plt.plot(val_loss_history, label='Validation Loss', color='orange')
        plt.title('Learning Curves')
        plt.xlabel('Epoch')
        plt.ylabel('KL Divergence Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/phase2_curves.png')
        plt.close()

        #early stopping and saving of the best model
        if best_val_error > val_loss:
            best_val_error = val_loss
            bad = 0
            models_path = models_dir / 'best.pt'
            torch.save({"encoder" : image_encoder.state_dict(), "decoder" : image_decoder.state_dict(), "attention" : image_attention.state_dict()}, models_path)
        else:
            bad += 1
            if bad == patience:
                break
