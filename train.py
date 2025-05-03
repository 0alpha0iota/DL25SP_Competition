import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import create_wall_dataloader
from jepa_model_2 import JEPA, save_model


def compute_loss(pred_latents, target_latents):
    return ((pred_latents - target_latents) ** 2).mean()


def train(
    epochs=2,
    lr=1e-4,
    batch_size=32,
    repr_dim=256,
    device="cuda",
):
    dataset = create_wall_dataloader(
        data_path="/scratch/DL25SP/train",
        probing=False,
        device=device,
        train=True,
        batch_size=batch_size
    )

    model = JEPA(repr_dim=repr_dim, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch in tqdm(dataset, desc=f"Epoch {epoch+1}/{epochs}"):
            states = batch.states.to(device)
            actions = batch.actions.to(device)

            pred_latents = model(states, actions)
            with torch.no_grad():
                target_latents = model.encode_targets(states)

            loss = compute_loss(pred_latents, target_latents)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model._update_target_encoder()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(dataset):.6f}")

    save_model(model, "jepa_checkpoint.pth")
    return model


if __name__ == "__main__":
    train()
