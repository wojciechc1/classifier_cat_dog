import torch.optim as optim
import torch.nn as nn

from model.cnn import CNN
from utils.data_loader import data_loader
from train import train


train_data_dir = "../Datasets/cats_and_dogs/cats_vs_dogs_split/train"
test_data_dir = "../Datasets/cats_and_dogs/cats_vs_dogs_split/val"

epochs = 5
learning_rate = 0.001
batch_size = 64


train_data, val_data = data_loader(train_data_dir, test_data_dir,  batch_size)

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    acc = train(model, criterion, optimizer, train_data)
    #TODO validate model and draw plots
    print("Accuracy: ", acc)