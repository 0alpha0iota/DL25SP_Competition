import torch
import torch.optim as optim
from tqdm import tqdm
import random      # for Gaussian noise
from jepa_model import JEPA
from dataset import create_wall_dataloader               

def train(
    epochs=10,
    lr=1e-4,
    batch_size=64,
    repr_dim=128,
    device="cuda",
):
    dataloader = JEPA().to(device) and create_wall_dataloader(
        data_path="/scratch/DL25SP/train",
        probing=False,
        device=device,
        train=True,
        batch_size=batch_size,
    )
    model = JEPA(repr_dim=repr_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            states, _, actions = batch.states, batch.next_states, batch.actions

            ### Gaussian noise augmentation
            if random.random() < 0.5:
                states = states + 0.01 * torch.randn_like(states)

            loss = model.compute_loss(states, actions)  ### use VICReg loss under the hood
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model._update_target_encoder()
            epoch_loss += loss.item()

        scheduler.step() 
        print(f"[train.py] Epoch {epoch+1}, avg loss={epoch_loss/len(dataloader):.6f}")

    model.save_model("jepa_checkpoint.pth")  # unified save API
    return model

if __name__ == "__main__":
    train()
