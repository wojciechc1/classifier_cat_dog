import torch
import torch.nn.functional as F

def train(model, criterion, optimizer, train_data):
    correct, total = 0, 0

    for i, (images, labels) in enumerate(train_data):

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        probabilities = F.softmax(outputs, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)

        correct += (predicted_classes == labels).sum().item()
        total += labels.size(0)

        if i % 10:
            print(correct / total)
    return correct / total