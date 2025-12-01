import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from q1 import HairTypeCNN  # Import model only!

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load previous training history
history = np.load('history_no_aug.npy', allow_pickle=True).item()

# Ensure val metrics exist (fix KeyError)
if 'val_loss' not in history:
    history['val_loss'] = []
if 'val_acc' not in history:
    history['val_acc'] = []

# Load model (same architecture, continued training)
model = HairTypeCNN().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.8)

# Dataset paths
data_dir = "./data"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "test")

# Training with augmentations
train_transforms = transforms.Compose([
    transforms.RandomRotation(50),
    transforms.RandomResizedCrop(200, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Validation stays normal
val_transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False)

num_epochs = 10
print("\nStarting fine-tuning with augmentations...\n")

for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0
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

    train_loss = running_loss / len(train_dataset)
    train_acc = correct_train / total_train

    history['loss'].append(train_loss)
    history['acc'].append(train_acc)

    # Validation
    model.eval()
    val_running_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss = val_running_loss / len(val_dataset)
    val_acc = correct_val / total_val

    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print(f"Epoch {10+epoch+1}/20 - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# After training — compute Q5 and Q6
mean_val_loss = float(np.mean(history['val_loss']))
last5_mean_acc = float(np.mean(history['val_acc'][-5:]))

print("\n===== RESULTS FOR Q5 & Q6 =====")
print("Mean Test Loss (Q5):", mean_val_loss)
print("Mean Test Accuracy Last 5 Epochs (Q6):", last5_mean_acc)

# Save updated history
np.save("history_aug.npy", history)
print("\nSaved updated history to history_aug.npy")

