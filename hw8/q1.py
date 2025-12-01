import torch
import torch.nn as nn

# Model definition according to homework instructions
class HairTypeCNN(nn.Module):
    def __init__(self):
        super(HairTypeCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 99 * 99, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # Optional: only runs when executing q1.py manually, NOT when imported!
    from torchsummary import summary
    import torch.optim as optim

    model = HairTypeCNN()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.8)

    print("Loss function:", criterion.__class__.__name__)
    summary(model, input_size=(3, 200, 200))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

