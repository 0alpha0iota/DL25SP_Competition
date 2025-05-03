import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from dataset import create_wall_dataloader


class ConvEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, output_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class MLPActionPredictor(nn.Module):
    def __init__(self, input_dim=128, action_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.fc(x)


class JEPA(nn.Module):
    def __init__(self, repr_dim=128):
        super().__init__()
        self.repr_dim = repr_dim
        self.encoder = ConvEncoder(repr_dim)
        self.target_encoder = ConvEncoder(repr_dim)
        self.predictor = MLPActionPredictor(repr_dim)

        self.temperature = 0.1

        self._init_target_encoder()

    def _init_target_encoder(self):
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def _update_target_encoder(self, m=0.99):
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

    def forward(self, states, actions):
        B, Tm1, _ = actions.shape
        reps = []
        s = self.encoder(states[:, 0])
        reps.append(s)

        for t in range(Tm1):
            s = self.predictor(s, actions[:, t])
            reps.append(s)

        return torch.stack(reps, dim=1)

    def compute_loss(self, states, actions):
        pred_states = self.forward(states, actions)
        with torch.no_grad():
            target_states = torch.stack([self.target_encoder(states[:, t]) for t in range(states.shape[1])], dim=1)

        pred_norm = F.normalize(pred_states.reshape(-1, pred_states.shape[-1]), dim=-1)
        target_norm = F.normalize(target_states.reshape(-1, target_states.shape[-1]), dim=-1)

        logits = (pred_norm @ target_norm.T) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)

        loss = F.cross_entropy(logits, labels)
        return loss


def train(epochs=2, batch_size=16, lr=1e-4, device="cuda"):
    dataloader = create_wall_dataloader(
        data_path="/scratch/DL25SP/train",
        probing=False,
        device=device,
        train=True,
        batch_size=batch_size
    )

    model = JEPA().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            states = batch.states.to(device)
            actions = batch.actions.to(device)

            loss = model.compute_loss(states, actions)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model._update_target_encoder()

            total_loss += loss.item()

            if i % 100 == 0:
                print(f"  Step {i}: loss={loss.item():.4f} | mem: {torch.cuda.memory_allocated() / 1e9:.2f}G")

        print(f"Epoch {epoch+1} avg loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "jepa_checkpoint.pth")
    return model

def load_model(path, device="cuda", repr_dim=128):
    model = JEPA(repr_dim=repr_dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

if __name__ == "__main__":
    train()







