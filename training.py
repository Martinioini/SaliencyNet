from pathlib import Path
import torch.nn.functional as F
import matplotlib
from metrics import cc_metric
from models import decoder, discriminator, encoder
matplotlib.use('Agg')  # backend non interattivo: plt.show() bloccherebbe lo script
import matplotlib.pyplot as plt
import numpy as np
import torch


# Training function, if phase1 the encoder is frozen
def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer, encoder_frozen=False):
    if encoder_frozen:
        encoder.eval()
    else :
        encoder.train()
    decoder.train()
    losses = []

    for image_batch, image_map_batch, image_fix_batch in dataloader:

        image_batch = image_batch.to(device)
        image_map_batch = image_map_batch.to(device)

        c1, c2, c3, c4 = encoder(image_batch)

        decoded_data = decoder(c1, c2, c3, c4)

        loss = loss_fn(decoded_data, image_map_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu().numpy())
    losses = np.mean(losses)
    return losses

#testing function, gradients are not kept and both models are in evak mode
def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    encoder.eval()
    decoder.eval()
    val_losses = []

    with torch.no_grad():

        for image_batch, image_map_batch, image_fix_batch in dataloader:
            image_batch = image_batch.to(device)
            image_map_batch = image_map_batch.to(device)

            c1, c2, c3, c4 = encoder(image_batch)
            decoded_data = decoder(c1, c2, c3, c4)

            loss = loss_fn(decoded_data, image_map_batch)
            val_losses.append(loss.item())

    return np.mean(val_losses)

#Phase 1: The encoder is frozen, only a few epochs of training are done
def run_phase1(image_encoder, image_decoder, device, train_loader, val_loader, loss_fn, optim1,
               train_loss_history, val_loss_history, num_epochs=5, patience=3):
    bad = 0
    best_val_error = float('inf')

    for p in image_encoder.parameters():
        p.requires_grad = False

    saved_models = Path.cwd() / "models"
    saved_models.mkdir(parents=True, exist_ok=True)
    saved_plots = Path.cwd() / "Phase 1 plots"
    saved_plots.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        print('EPOCH %d/%d' % (epoch + 1, num_epochs))
        ### Training
        train_loss = train_epoch(
            encoder=image_encoder,
            decoder=image_decoder,
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
            device=device,
            dataloader=val_loader,
            loss_fn=loss_fn)

        # Print Validation curves
        print(f'VALIDATION - EPOCH {epoch+1}/{num_epochs} - loss: {val_loss}\n')
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        path_graph = saved_plots / f"phase1_curves_{epoch+1}.png"
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_history, label='Train Loss', color='blue')
        plt.plot(val_loss_history, label='Validation Loss', color='orange')
        plt.title('Learning Curves')
        plt.xlabel('Epoch')
        plt.ylabel('KL Divergence Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(path_graph)
        plt.close()

        #early stopping and saving of the best model
        if best_val_error > val_loss:
            best_val_error = val_loss
            bad = 0
            path_model = saved_models / "phase1.pt"
            torch.save({"encoder" : image_encoder.state_dict(), "decoder" : image_decoder.state_dict()}, path_model)
        else:
            bad += 1
            if bad == patience:
                break 

#Phase 2: The whole network is used
def run_phase2(image_encoder, image_decoder, device, train_loader, val_loader, loss_fn, optim, scheduler,
               train_loss_history, val_loss_history, num_epochs=50, patience=3):
    bad = 0
    best_val_error = float('inf')
    #load the best model from phase 1
    model = torch.load(Path.cwd() / 'models' / 'phase1.pt', map_location=device, weights_only=False)
    image_encoder.load_state_dict(model['encoder'])
    image_decoder.load_state_dict(model['decoder'])
    image_encoder.train()
    image_decoder.train()

    #unfreeze the encoder and train on the whole net
    for p in image_encoder.parameters():
        p.requires_grad = True
    saved_models = Path.cwd() / "models"
    saved_models.mkdir(parents=True, exist_ok=True)
    saved_plots = Path.cwd() / "Phase 2 plots"
    saved_plots.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        print('EPOCH %d/%d' % (epoch + 1, num_epochs))
        ### Training (use the training function)
        train_loss = train_epoch(
            encoder=image_encoder,
            decoder=image_decoder,
            device=device,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optim)
        print(f'TRAIN - EPOCH {epoch+1}/{num_epochs} - loss: {train_loss}')

        ### Validation  (use the testing function)
        val_loss = test_epoch(
            encoder=image_encoder,
            decoder=image_decoder,
            device=device,
            dataloader=val_loader,
            loss_fn=loss_fn)

        scheduler.step(val_loss)

        # Print Validation curves
        print(f'VALIDATION - EPOCH {epoch+1}/{num_epochs} - loss: {val_loss}\n')
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        path_graph = saved_plots / f"phase2_curves_{epoch+1}.png"
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_history, label='Train Loss', color='blue')
        plt.plot(val_loss_history, label='Validation Loss', color='orange')
        plt.title('Learning Curves')
        plt.xlabel('Epoch')
        plt.ylabel('KL Divergence Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(path_graph)
        plt.close()

        #early stopping and saving of the best model
        if best_val_error > val_loss:
            best_val_error = val_loss
            bad = 0
            path_model = saved_models / "phase2.pt"
            torch.save({"encoder" : image_encoder.state_dict(), "decoder" : image_decoder.state_dict()}, path_model)
        else:
            bad += 1
            if bad == patience:
                break


def discriminator_training(encoder, decoder, discriminator, device, dataloader, loss_fn, optimizer):
    for param in encoder.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False
    encoder.eval()
    decoder.eval()
    discriminator.train()
    losses = []

    for image_batch, image_map_batch, image_fix_batch in dataloader:

        image_batch = image_batch.to(device)
        image_map_batch = image_map_batch.to(device)

        c1, c2, c3, c4 = encoder(image_batch)
        fake_image_batch = decoder(c1, c2, c3, c4).detach()

        real_sample = discriminator(image_batch, image_map_batch)    
        fake_data = discriminator(image_batch, fake_image_batch)

        loss = loss_fn(real_sample, torch.ones_like(real_sample)) + loss_fn(fake_data, torch.zeros_like(fake_data))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu().numpy())
    losses = np.mean(losses)
    return losses

def discriminator_testing(encoder, decoder, discriminator, device, dataloader, loss_fn):
    encoder.eval()
    decoder.eval()
    discriminator.eval()
    val_losses = []
    true_positive = 0
    false_positive = 0
    total = 0

    with torch.no_grad():

        for image_batch, image_map_batch, image_fix_batch in dataloader:

            image_batch = image_batch.to(device)
            image_map_batch = image_map_batch.to(device)

            c1, c2, c3, c4 = encoder(image_batch)
            fake_image_batch = decoder(c1, c2, c3, c4).detach()

            real_sample = discriminator(image_batch, image_map_batch)    
            fake_data = discriminator(image_batch, fake_image_batch)
            
            true_positive += (real_sample > 0.5).sum().item()
            false_positive += (fake_data < 0.5).sum().item()
            total += real_sample.size(0)

            loss = loss_fn(real_sample, torch.ones_like(real_sample)) + loss_fn(fake_data, torch.zeros_like(fake_data))

            val_losses.append(loss.detach().cpu().numpy())

    print(f"True positive accuracy = {true_positive / total} False positive: {false_positive/ total}")

    return np.mean(val_losses)




#Phase 3: The generator is frozen, discriminator is trained for a few epochs (warm-up)
def run_phase3(image_encoder, image_decoder, discriminator, device, train_loader, val_loader, loss_fn, optim,
               train_loss_history, val_loss_history, num_epochs=5, patience=3):
    bad = 0
    best_val_error = float('inf')

    for p in image_encoder.parameters():
        p.requires_grad = False
    saved_models = Path.cwd() / "models"
    saved_models.mkdir(parents=True, exist_ok=True)
    saved_plots = Path.cwd() / "Phase 3 plots"
    saved_plots.mkdir(parents=True, exist_ok=True)


    for epoch in range(num_epochs):
        print('EPOCH %d/%d' % (epoch + 1, num_epochs))
        ### Training
        train_loss = discriminator_training(
            encoder=image_encoder,
            decoder=image_decoder,
            discriminator=discriminator,
            device=device,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optim)
        print(f'TRAIN - EPOCH {epoch+1}/{num_epochs} - loss: {train_loss}')

        ### Validation
        val_loss = discriminator_testing(
            encoder=image_encoder,
            decoder=image_decoder,
            discriminator=discriminator,
            device=device,
            dataloader=val_loader,
            loss_fn=loss_fn)

        # Print Validation curves
        print(f'VALIDATION - EPOCH {epoch+1}/{num_epochs} - loss: {val_loss}\n')
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        path_graph = saved_plots / f"phase3_curves_{epoch+1}.png"
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_history, label='Train Loss', color='blue')
        plt.plot(val_loss_history, label='Validation Loss', color='orange')
        plt.title('Learning Curves')
        plt.xlabel('Epoch')
        plt.ylabel('BCE Divergence Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(path_graph)
        plt.close()

        #early stopping and saving of the best model
        if best_val_error > val_loss:
            best_val_error = val_loss
            bad = 0
            path_model = saved_models / "phase3.pt"
            torch.save({"discriminator" : discriminator.state_dict()}, path_model)
        else:
            bad += 1
            if bad == patience:
                break



def adversarial_training(encoder, decoder, discriminator, device, dataloader, loss_fn, optimizer_generator, optimizer_discriminator, alpha=0.005):
    
    losses_discriminator = []
    losses_generator = []
    #just to be sure, since adversarial training is done after phase 3 (discriminiator warm-up), the encoder and decoder should be unfrozen
    for p in encoder.parameters():       p.requires_grad = True
    for p in decoder.parameters():       p.requires_grad = True
    for p in discriminator.parameters(): p.requires_grad = True
    
    #changed both encoder and decoder to train mode (before they were both on eval).
    encoder.train()
    decoder.train()
    discriminator.train()

    for image_batch, image_map_batch, image_fix_batch in dataloader:
        
        image_batch = image_batch.to(device)
        image_map_batch = image_map_batch.to(device)

        c1, c2, c3, c4 = encoder(image_batch)
        #added sigmoid here
        fake_image_batch = torch.sigmoid(decoder(c1, c2, c3, c4).detach())

        real_sample = discriminator(image_batch, image_map_batch)    
        fake_data = discriminator(image_batch, fake_image_batch)
        loss_discriminator = loss_fn(real_sample, torch.ones_like(real_sample)) + loss_fn(fake_data, torch.zeros_like(fake_data))

        optimizer_discriminator.zero_grad()
        loss_discriminator.backward()
        optimizer_discriminator.step()
        losses_discriminator.append(loss_discriminator.detach().cpu().numpy())

        c1, c2, c3, c4 = encoder(image_batch)
        #added sigmoid here
        fake_image_batch = torch.sigmoid(decoder(c1, c2, c3, c4))

        #computing the CC metric in the training is a mistake since the weights are constantly changing so the CC metric is unreliable.
        #These 4 lines shall be deleted.    
        #B = fake_image_batch.size(0)
        #pred = F.softmax(fake_image_batch.view(B, -1), dim=1).view_as(fake_image_batch)   #Same map as eval loop
        #cc_score = cc_metric(pred, image_map_batch).mean()   
        #cc.append(cc_score)

        #deleted the following line: "real_sample = discriminator(image_batch, image_map_batch)", since it was already computed above and is not needed again.
        #intuitively the generator loss should be computed using the discriminator output on the real samples and the fake samples 
        # (generator only wants to fool the discriminator, generator does not care what discriminator thinks about the real map), which we already have above.
        fake_data = discriminator(image_batch, fake_image_batch)
        loss_generator = alpha * loss_fn(fake_image_batch, image_map_batch) + loss_fn(fake_data, torch.ones_like(fake_data))

        optimizer_generator.zero_grad()
        loss_generator.backward()
        #when this code will run we should see a number > 0, otherwise there is probably a 'detach' somwehere preventing the gradients to flow back.
        print(encoder.layer4[0].conv1.weight.grad.norm())
        optimizer_generator.step()
        losses_generator.append(loss_generator.detach().cpu().numpy())
        
    return np.mean(losses_generator), np.mean(losses_discriminator)

def adversarial_testing(encoder, decoder, discriminator, device, dataloader, loss_fn, alpha=0.005):
    
    cc = []
    losses_discriminator = []
    losses_generator = []
    for p in encoder.parameters():       p.requires_grad = False
    for p in decoder.parameters():       p.requires_grad = False
    for p in discriminator.parameters(): p.requires_grad = False
    
    encoder.eval()
    decoder.eval()
    discriminator.eval()

    with torch.no_grad():

        for image_batch, image_map_batch, image_fix_batch in dataloader:
            
            image_batch = image_batch.to(device)
            image_map_batch = image_map_batch.to(device)

            c1, c2, c3, c4 = encoder(image_batch)
            fake_image_batch = torch.sigmoid(decoder(c1, c2, c3, c4)).detach()

            real_sample = discriminator(image_batch, image_map_batch)    
            fake_data = discriminator(image_batch, fake_image_batch)

            loss_discriminator = loss_fn(real_sample, torch.ones_like(real_sample)) + loss_fn(fake_data, torch.zeros_like(fake_data))
            losses_discriminator.append(loss_discriminator.detach().cpu().numpy())

            cc_score = cc_metric(fake_image_batch, image_map_batch).mean()   
            cc.append(cc_score)

            loss_generator = alpha * loss_fn(fake_image_batch, image_map_batch) + loss_fn(fake_data, torch.ones_like(fake_data))

            losses_generator.append(loss_generator.detach().cpu().numpy())

    return np.mean(cc), np.mean(losses_generator), np.mean(losses_discriminator)

#Phase 4: Both the generator and discriminator are trained in an adversarial way, for a few epochs
# Needs work, only added both optim instead of one in the firm, changed the name of the saved model and the plot folder.
def run_phase4(image_encoder, image_decoder, discriminator, device, train_loader, val_loader, loss_fn, optim_generator, optim_discriminator,
               train_loss_history, val_loss_history, num_epochs=5, patience=3):
    bad = 0
    best_val_error = float('inf')

    for p in image_encoder.parameters():
        p.requires_grad = False
    saved_models = Path.cwd() / "models"
    saved_models.mkdir(parents=True, exist_ok=True)
    saved_plots = Path.cwd() / "Phase 4 plots"
    saved_plots.mkdir(parents=True, exist_ok=True)


    for epoch in range(num_epochs):
        print('EPOCH %d/%d' % (epoch + 1, num_epochs))
        ### Training
        train_loss = discriminator_training(
            encoder=image_encoder,
            decoder=image_decoder,
            discriminator=discriminator,
            device=device,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optim_generator=optim_generator,
            optim_discriminator=optim_discriminator)
        print(f'TRAIN - EPOCH {epoch+1}/{num_epochs} - loss: {train_loss}')

        ### Validation
        val_loss = discriminator_testing(
            encoder=image_encoder,
            decoder=image_decoder,
            discriminator=discriminator,
            device=device,
            dataloader=val_loader,
            loss_fn=loss_fn)

        # Print Validation curves
        print(f'VALIDATION - EPOCH {epoch+1}/{num_epochs} - loss: {val_loss}\n')
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        path_graph = saved_plots / f"phase4_curves_{epoch+1}.png"
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_history, label='Train Loss', color='blue')
        plt.plot(val_loss_history, label='Validation Loss', color='orange')
        plt.title('Learning Curves')
        plt.xlabel('Epoch')
        plt.ylabel('BCE Divergence Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(path_graph)
        plt.close()

        #early stopping and saving of the best model
        if best_val_error > val_loss:
            best_val_error = val_loss
            bad = 0
            path_model = saved_models / "phase4.pt"
            torch.save({"discriminator" : discriminator.state_dict()}, path_model)
        else:
            bad += 1
            if bad == patience:
                break
