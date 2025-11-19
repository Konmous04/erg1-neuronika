import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import matplotlib.pyplot as plt


class simpleMLP(nn.Module):
    def __init__(self, h1=2048, h2=2048, h3=2048, h4=2048, h5=2048, h6=2048):
        super(simpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 32 * 32, h1)
        self.relu1 = nn.ReLU()
    
        self.fc2 = nn.Linear(h1, h2)
        self.relu2 = nn.ReLU()
    
        self.fc3 = nn.Linear(h2, h3)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(h3, h4)
        self.relu4 = nn.ReLU()

        self.fc5 = nn.Linear(h4, h5)
        self.relu5 = nn.ReLU()

        self.fc6 = nn.Linear(h5, h6)
        self.relu6 = nn.ReLU()

        self.fc7 = nn.Linear(h6, 10)

    def forward(self, x):
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        x = self.relu3(x)

        x = self.fc4(x)
        x = self.relu4(x)

        x = self.fc5(x)
        x = self.relu5(x)

        x = self.fc6(x)
        x = self.relu6(x)

        x = self.fc7(x)
        
        return x

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()*images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss/total
    epoch_accuracy = correct/total
    return epoch_loss, epoch_accuracy

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():    
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()*images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy

if __name__ == "__main__":

    train_list_loss = []
    train_list_acc = []
    test_list_loss = []
    test_list_acc = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


    model = simpleMLP(h1=2048, h2=2048, h3=2048, h4=2048, h5=2048, h6=2048).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    for epoch in range(num_epochs):
        
        start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        end = time.perf_counter()
        total_time = end-start
        
        train_list_loss.append(train_loss)
        train_list_acc.append(train_acc)
        test_list_loss.append(test_loss)
        test_list_acc.append(test_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f} | "
              f"Time: {total_time:.1f}sec"
             )
        
    x1 = range(len(train_list_loss))
    x2 = range(len(train_list_acc))
    x3 = range(len(test_list_loss))
    x4 = range(len(test_list_acc))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 8))

    ax1.plot(x1, train_list_loss)
    ax1.set_title("Train Loss")

    ax2.plot(x2, train_list_acc)
    ax2.set_title("Train Accuracy")

    ax3.plot(x3, test_list_loss)
    ax3.set_title("Test Loss")

    ax4.plot(x4, test_list_acc)
    ax4.set_title("Test Accuracy")

    plt.tight_layout()
    plt.show()