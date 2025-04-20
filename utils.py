from torchvision import transforms
from torch.utils.data import Dataset
import os
import PIL
from typing import List, Tuple
import matplotlib.pyplot as plt

class TrainDataset(Dataset):
    def __init__(self, images, labels):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ])
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = PIL.Image.open(image_path)
        image = self.transform(image)
        label = self.labels[idx]
        return image, label

class TestDataset(Dataset):
    def __init__(self, images):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = PIL.Image.open(image_path)
        image = self.transform(image)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return image, base_name

def load_train_dataset(path='data/train/') -> Tuple[List, List]:
    images = []
    labels = []
    class_mapping = {'elephant': 0, 'jaguar': 1, 'lion': 2, 'parrot': 3, 'penguin': 4}
    for class_name, label in class_mapping.items():
        class_dir = os.path.join(path, class_name)
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_file)
                images.append(img_path)
                labels.append(label)
    return images, labels

def load_test_dataset(path='data/test/') -> List:
    images = []
    for img_file in os.listdir(path):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(path, img_file)
            images.append(img_path)
    return images

def plot(train_losses: List, val_losses: List):
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.close()
    print("Save the plot to 'loss.png'")