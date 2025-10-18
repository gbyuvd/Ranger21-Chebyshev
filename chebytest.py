import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np, random, time, matplotlib.pyplot as plt

# ====================================================
# Imports
# ====================================================
from ranger21 import Ranger21
from ranger21.chebyshev_lr_functions import ChebyshevLR, get_chebs, get_cheb_lr

# ====================================================
# Setup
# ====================================================
seed = 2025
torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using {device}")

# ====================================================
# Dataset (subset of sklearn digits)
# ====================================================
data = load_digits()
X = torch.tensor(data.images, dtype=torch.float32).unsqueeze(1) / 16.0
y = torch.tensor(data.target, dtype=torch.long)
X, y = X[:500], y[:500]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True, num_workers=0)
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False, num_workers=0)

# ====================================================
# Tiny CNN Model
# ====================================================
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

# ====================================================
# Ranger21 Optimizer (with Chebyshev)
# ====================================================
num_epochs = 5  # extend to see clearer LR dynamics
num_batches = len(train_loader)
model = TinyCNN().to(device)

optimizer_params = {
    'lr': 1e-3,
    'weight_decay': 0.01,
    'use_adabelief': True,
    'use_cheb': False,
    'use_warmup': True,
    'use_madgrad': True,
    'num_epochs': num_epochs,
    'using_gc': True,
    'warmdown_active': False,
    'num_batches_per_epoch': num_batches
}
optimizer = Ranger21(model.parameters(), **optimizer_params)
# --- External Scheduler ---
total_steps = num_epochs * num_batches
cheb_scheduler = ChebyshevLR(
    optimizer,
    total_steps=total_steps,
    lr_start=1e-3,
    lr_end=1e-5,
    min_lr=1e-6
)
# ====================================================
# Train and record Ranger21 LR evolution
# ====================================================
criterion = nn.CrossEntropyLoss()
lrs = []

for epoch in range(num_epochs):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        cheb_scheduler.step()  # update LR schedule externally

        # record LR
        lrs.append(optimizer.param_groups[0]["lr"])

# ====================================================
# Compute theoretical Chebyshev curve for contrast
# ====================================================
total_steps = ChebyshevLR.total_steps_from(num_epochs, num_batches)
sched = get_chebs(num_epochs, num_batches)
cheb_curve = [1e-3 * get_cheb_lr(1.0, i, num_batches, num_epochs) for i in range(total_steps)]

# ====================================================
# Plot Comparison
# ====================================================
plt.figure(figsize=(9,4))
plt.plot(lrs, label="Ranger21 internal LR", lw=2)
plt.plot(cheb_curve, "--", label="Theoretical Chebyshev LR", lw=2)
plt.title("Comparison: Ranger21 with External Sched vs. Theoretical Chebyshev Schedule")
plt.xlabel("Training Step")
plt.ylabel("Learning Rate")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()
