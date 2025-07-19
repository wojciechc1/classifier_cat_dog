from utils.data_loader import data_loader
from PIL import Image, UnidentifiedImageError
import torch

train_data_dir = "../Datasets/cats_and_dogs/cats_vs_dogs_split/train"
test_data_dir = "../Datasets/cats_and_dogs/cats_vs_dogs_split/val"


train_data, val_data = data_loader(train_data_dir, test_data_dir,  2)

print(len(train_data))


for i, (images, labels) in enumerate(train_data):
    print(images.shape)
    #images = images.reshape(images.size(0), -1)
    print(images)
    if i == 0:
        break