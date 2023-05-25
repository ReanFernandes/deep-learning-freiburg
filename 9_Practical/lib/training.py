"""Training of models"""

import torch
import torch.nn as nn
import torch.optim as optim

from lib.utils import create_dataloaders

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Running on device: {device}")


def train_and_evaluate(net, num_epochs=2) -> None:
    """Train and evaluate the given model

    Args:
        net: CNN model to train
        num_epochs: Number of epochs to train

    """
    net = net.to(device)

    # Define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Get dataloaders
    trainloader, testloader = create_dataloaders()

    net.train()
    # Train the network
    for epoch in range(num_epochs):

        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            # print(inputs.size())
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 5 == 4:    # print every 5 mini-batches
                print('[epoch %d, mini-batch #%3d] loss: %.6f' % (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0

    print('Finished Training')

    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on test images: %d %%' % (
        100 * correct / total))
