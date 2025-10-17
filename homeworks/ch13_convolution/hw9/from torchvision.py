from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torch import nn, optim

root = "/Users/wangliwei/Desktop/hw9/output_split"  # 改成你的路径
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

tfm = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
train_ds = ImageFolder(f"{root}/train", transform=tfm)
val_ds   = ImageFolder(f"{root}/val",   transform=tfm)
test_ds  = ImageFolder(f"{root}/test",  transform=tfm)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

val_loader   = DataLoader(val_ds, batch_size=16)
test_loader  = DataLoader(test_ds, batch_size=16)

num_classes = len(train_ds.classes)

# 一个很小的 CNN
model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2),                 # 112x112
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),         # -> (B, 32, 1, 1)
    nn.Flatten(),
    nn.Linear(32, num_classes)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def evaluate(loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += criterion(logits, y).item() * y.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return loss_sum/total, correct/total

# 训练 5 轮
for epoch in range(1, 6):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    val_loss, val_acc = evaluate(val_loader)
    print(f"Epoch {epoch}: val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

test_loss, test_acc = evaluate(test_loader)
print("Test:", test_loss, test_acc, "Classes:", train_ds.classes)
