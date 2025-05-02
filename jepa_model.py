import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random                   ### for random-horizon rollouts
from tqdm import tqdm
from dataset import create_wall_dataloader

### Simple residual block to boost representational power
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class ConvEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, 4, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            ResidualBlock(16),  ### residual block
            nn.Conv2d(16, 32, 4, stride=2, padding=1), # 32 -> 16
            nn.ReLU(),
            ResidualBlock(32),  ### residual block
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # 16 -> 8
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
            nn.Dropout(0.1),
            nn.Linear(256, input_dim),
        )

    def forward(self, state, action):
        return self.fc(torch.cat([state, action], dim=-1))


class JEPA(nn.Module):
    def __init__(self, repr_dim=128):
        super().__init__()
        self.repr_dim = repr_dim
        self.encoder = ConvEncoder(repr_dim)
        self.target_encoder = ConvEncoder(repr_dim)
        self.predictor = MLPActionPredictor(repr_dim)
        self._init_target_encoder()

    def _init_target_encoder(self):
        for q, k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            k.data.copy_(q.data); k.requires_grad = False

    def _update_target_encoder(self, m=0.99):
        for q, k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            k.data.mul_(m).add_(q.data, alpha=1-m)

    def encode_targets(self, states):
        ### expose for train.py compatibility
        with torch.no_grad():
            return torch.stack([
                self.target_encoder(states[:, t])
                for t in range(states.size(1))
            ], dim=1)

    def forward(self, states, actions, rollout_len=None):
        """
        states: [B, T, 2, 64, 64]
        actions: [B, T-1, 2]
        rollout_len: optional int to truncate horizon randomly
        """
        B, T, _, _, _ = states.shape
        max_h = T-1 if rollout_len is None else rollout_len
        s = self.encoder(states[:, 0])
        reps = [s]
        for t in range(max_h):
            s = self.predictor(s, actions[:, t])
            reps.append(s)
        return torch.stack(reps, dim=1)

    def compute_vicreg_loss(self, pred, target,
                            lambda_invariance=25.0, lambda_variance=25.0, lambda_covariance=1.0,
                            eps=1e-4):
        # 1) Invariance: MSE between normalized vectors
        pred_norm = F.normalize(pred, dim=-1)
        target_norm = F.normalize(target, dim=-1)
        inv_loss = F.mse_loss(pred_norm, target_norm)

        # 2) Variance: encourage std >= 1 per feature
        def variance_term(x):
            std = torch.sqrt(x.var(dim=0) + eps)
            return torch.mean(F.relu(1 - std))
        var_loss = variance_term(pred_norm) + variance_term(target_norm)

        # 3) Covariance: off-diagonal covariance
        def covariance_term(x):
            N, D = x.size()
            x = x - x.mean(dim=0)
            cov = (x.T @ x) / (N - 1)
            off_diag = cov.flatten()[~torch.eye(D, dtype=bool, device=cov.device)]
            return (off_diag ** 2).sum() / D
        cov_loss = covariance_term(pred_norm) + covariance_term(target_norm)

        return (lambda_invariance * inv_loss +
                lambda_variance   * var_loss +
                lambda_covariance * cov_loss)

    def compute_loss(self, states, actions):
        ### Random-horizon curriculum
        rollout_len = random.randint(5, states.size(1)-1)
        pred = self.forward(states, actions, rollout_len)
        target = self.encode_targets(states)[:, :rollout_len+1]
        # reshape to [B*(H+1), D]
        B, H1, D = pred.shape
        pred_flat   = pred.reshape(-1, D)
        target_flat = target.reshape(-1, D)
        ### VICReg collapse-prevention loss
        return self.compute_vicreg_loss(pred_flat, target_flat)

    def train_model(self, epochs=10, batch_size=64, lr=1e-4, device="cuda"):
        dataloader = create_wall_dataloader(
            data_path="/scratch/DL25SP/train",
            probing=False,
            device=device,
            train=True,
            batch_size=batch_size,
        )
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        ### CosineAnnealingLR scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        self.to(device)
        for epoch in range(epochs):
            total_loss = 0
            for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
                states, _, actions = batch.states, batch.next_states, batch.actions
                ### Gaussian noise augmentation
                if random.random() < 0.5:
                    states = states + 0.01 * torch.randn_like(states)
                loss = self.compute_loss(states, actions)
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                self._update_target_encoder()
                total_loss += loss.item()
            scheduler.step()  # --- MODIFIED
            print(f"[JEPA Train] Epoch {epoch+1}, avg loss={total_loss/len(dataloader):.4f}")

    def save_model(self, path="jepa_checkpoint.pth"):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load_model(path, device="cuda", repr_dim=128):
        model = JEPA(repr_dim).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model

if __name__ == "__main__":
    jepa = JEPA()
    jepa.train_model()
    jepa.save_model()
