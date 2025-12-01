import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from q1 import HairTypeCNN  # Reuse the model from q1.py

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Paths to data folders
data_dir = "./data"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "test")

# Transformations without augmentations
train_transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=train_transforms)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False)

# Model + loss + optimizer
model = HairTypeCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.8)

# Training loop
history = {'acc': [], 'loss': []}

num_epochs = 10
print("\nStarting training for Q3 & Q4...\n")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct_train / total_train
    history['loss'].append(epoch_loss)
    history['acc'].append(epoch_acc)

    print(f"Epoch {epoch+1}/{num_epochs}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.4f}")

# Compute Q3 & Q4 stats
median_train_acc = float(np.median(history['acc']))
std_train_loss = float(np.std(history['loss']))

print("\n----- RESULTS FOR Q3 & Q4 -----")
print(f"Median Train Accuracy (Q3): {median_train_acc}")
print(f"STD Train Loss (Q4): {std_train_loss}")

# Save history for augmentation stage
np.save('history_no_aug.npy', history)
print("\nSaved history to history_no_aug.npy")

