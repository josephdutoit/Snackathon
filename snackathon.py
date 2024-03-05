"""Fine-tune a ResNet for image classification"""


# !pip install -q torch torchvision
# !nvidia-smi
from torchvision.models import resnet152
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import torch
from torch import optim, nn
import zipfile
import os
import sys
from PIL import Image, ImageOps
from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import gc
import yaml


#########################
# Import data
#########################

# CODE WHEN USING COLAB
# from google.colab import files
# from google.colab import drive
# drive.mount('/content/drive')
# !cp /content/drive/MyDrive/hackathon_data.zip /content
# !unzip /content/hackathon_data.zip

with open('bev_classification/names.yaml', 'r') as f:
    LABEL_LIST = yaml.full_load(f)['classes']

LABEL_DICT = dict([(LABEL_LIST[i], i) for i in range(len(LABEL_LIST))])


#############################
# Implement Dataset class
#############################

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, train=False):
        self.transform = transform
        self.data_dir = '/home/jcdutoit/Snackathon/bev_classification'
        # self.data_labels = os.path.join(self.data_dir, "/datasets/train.txt")
        if train:
          self.data_labels = "/home/jcdutoit/Snackathon/bev_classification/datasets/train.txt"
          raw_txt = np.loadtxt(self.data_labels, dtype=str)
          lines_labels = np.array([line.split(',') for line in raw_txt])
        else:
          self.data_labels = "/home/jcdutoit/Snackathon/bev_classification/datasets/test_edited.txt"
          raw_txt = np.loadtxt(self.data_labels, dtype=str)
          lines_labels = np.array([[line, line.split('/')[2]] for line in raw_txt])

        self.df = pd.DataFrame(lines_labels, columns=['image', 'label'])
        self.transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.PILToTensor()])

        # potential TODO: augment data with transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the image and label from the file
        img_path = os.path.join(self.data_dir, self.df.iloc[idx, 0])
        image = Image.open(img_path)
        label = LABEL_DICT[self.df.iloc[idx, 1]]
        l_tensor = torch.tensor(label)

        # Convert the label to one hot encoding (for non-numeric labels)
        # one_hot = torch.zeros(len(self.label_dict))
        # one_hot[label] = 1.0

        # Transform the image
        image_t = self.transform(image).float()
        return image_t, l_tensor


##########################################################
# Implement pre-trained ResNet and wrap as nn.Module
##########################################################

class CustomResNet(nn.Module):
    def __init__(self, num_classes, start_frozen=False):
        super(CustomResNet, self).__init__()

        # Load the model - make sure it is pre-trained
        self.res_model = resnet152(pretrained=True)

        if start_frozen:
            # Turn off all gradients of the resnet
            for param in self.res_model.parameters():
                param.requires_grad = False

        # Override the output linear layer of the neural network to map to the correct number of classes. Note that this new layer has requires_grad = True
        self.res_model.fc = nn.Linear(self.res_model.fc.in_features, 99)

    def unfreeze(self, n_layers):
        # Turn on gradients for the last n_layers
        child_list = list(self.res_model.children())
        for child in child_list[len(child_list) - n_layers - 1:]:
           for param in child.parameters():
               param.requires_grad = True

    def forward(self, x):
        # Pass x through the resnet
        return self.res_model(x)


########################
# Training Loop
########################

def accuracy(y_hat, y_truth):
    """Gets average accuracy of a vector of predictions"""
    preds = torch.argmax(y_hat, dim=1)
    acc = torch.mean((preds == y_truth).float())
    return acc


def evaluate(model, objective, val_loader, device, epoch):
    """Gets average accuracy and loss for the validation set"""
    val_losses = 0
    val_accs = 0
    batches = 0
    # model.eval() so that batchnorm and dropout work in eval mode
    model.eval()
    # torch.no_grad() to turn off computation graph creation. This allows for temporal
    # and spatial complexity improvements, which allows for larger validation batch
    # sizes so it’s recommended
    preds = []
    top5_preds = []

    with torch.no_grad():
        for x, y_truth in val_loader:

            batches += 1

            x, y_truth = x.to(device), y_truth.to(device)
            y_hat = model(x)
            val_loss = objective(y_hat, y_truth)
            val_acc = accuracy(y_hat, y_truth)
            preds.append(LABEL_LIST[int(torch.argmax(y_hat, dim=1).item())])
            
            # Do top 5
            _, ind = y_hat.topk(5, dim=1, largest=True)
            ind = ind.tolist()[0]
            top5_preds.append([LABEL_LIST[int(ind[i])] for i in range(len(ind))])

            val_losses += val_loss.item()
            val_accs += val_acc

    print("Writing to /home/jcdutoit/Snackathon/val_" + str(epoch) + ".txt'")
    with open('/home/jcdutoit/Snackathon/val_'+str(epoch)+'.txt', 'w') as f:
        for pred in preds:
            f.write(pred + '\n')

    with open('/home/jcdutoit/Snackathon/top5_val_'+str(epoch)+'.txt', 'w') as f:
        for idx in top5_preds:
            pred_string = ', '.join(idx)
            f.write(pred_string + '\n')

    model.train()

    return val_losses/batches, val_accs/batches


def train(start_frozen=False, model_unfreeze=0):
    """Fine-tunes a CNN
    Args:
        start_frozen (bool): whether to start with the network weights frozen.
        model_unfreeze (int): the maximum number of network layers to unfreeze
    """
    gc.collect()
    epochs = 5
    # Start with a very low learning rate
    lr = .00005
    val_every = 500
    num_classes = 16
    batch_size = 32
    device = torch.device('cuda:0')

    # Data
    train_dataset = CustomDataset(train=True)
    val_dataset = CustomDataset(train=False)

    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              num_workers=8,
                              batch_size=batch_size)
    val_loader = DataLoader(val_dataset,
                              shuffle=False,
                              num_workers=8,
                              batch_size=1)

    # Model
    model = CustomResNet(num_classes, start_frozen=start_frozen).to(device)

    # Objective
    objective = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-1)

    # Progress bar
    pbar = tqdm(total=len(train_loader) * epochs)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    cnt = 0
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        # Implement model unfreezing
        if cnt < model_unfreeze and cnt % 45 == 0:
            # Part 1.4
            # Unfreeze the last layers, one more each epoch
            layers = int(cnt / 45)
            print("\nUnfreezing " + str(layers) + " layers")
            model.unfreeze(layers)

        for x, y_truth in train_loader:

            x, y_truth = x.to(device), y_truth.to(device)

            optimizer.zero_grad()

            y_hat = model(x)
            train_loss = objective(y_hat, y_truth)
            train_acc = accuracy(y_hat, y_truth)

            train_loss.backward()
            optimizer.step()

            train_accs.append(train_acc)
            train_losses.append(train_loss.item())

            if cnt % val_every == 0:
                val_loss, val_acc = evaluate(model, objective, val_loader, device, cnt)
                val_losses.append(val_loss)
                val_accs.append(val_acc)

            pbar.set_description('train loss:{:.4f}, train accuracy:{:.4f}, val loss:{:.4f}, val accuracy:{:.4f}.'.format(train_loss.item(), 
                                                                                                                          train_acc, 
                                                                                                                          val_losses[-1], 
                                                                                                                          val_accs[-1]))
            pbar.update(1)
            cnt += 1

    pbar.close()
    torch.save(model, '/home/jcdutoit/Snackathon/snack_model.pt')

    # Plot results
    # plt.subplot(121)
    # plt.plot(np.arange(len(train_accs)), train_accs, label='Train Accuracy')
    # plt.plot(np.arange(len(train_accs), step=val_every), val_accs, label='Val Accuracy')
    # plt.legend()
    # plt.subplot(122)
    # plt.plot(np.arange(len(train_losses)), train_losses, label='Train Loss')
    # plt.plot(np.arange(len(train_losses), step=val_every), val_losses, label='Val Loss')
    # plt.legend()
    # plt.show()


########################################
# Use saved model to make predictions
########################################

class PredictionDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, train=False):
        self.transform = transform
        self.data_dir = '/home/jcdutoit/Snackathon/bev_classification'
        self.data_labels = "/home/jcdutoit/Snackathon/bev_classification/datasets/test_edited.txt"
        raw_txt = np.loadtxt(self.data_labels, dtype=str)
        lines = np.array([line for line in raw_txt])
        self.df = pd.DataFrame(lines, columns=['image'])
        self.transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.PILToTensor()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the image from the file
        img_path = os.path.join(self.data_dir, self.df.iloc[idx, 0])
        image = Image.open(img_path)

        # Transform the image
        image_t = self.transform(image).float()
        return image_t
    

def predict(model, test_loader, device):
    """Runs data through saved model and writes predictions to .txt files"""
    # Store predicted labels
    preds1 = []
    preds5 = []

    with torch.no_grad():
        for x in test_loader:
            x = x.to(device)
            y_hat = model(x)

            # Find most likely label
            preds1.append(LABEL_LIST[int(torch.argmax(y_hat, dim=1).item())])
            # Find top 5 most likely for each image
            top5_indices = torch.topk(y_hat, 5, dim=1, largest=True)
            row = []
            for index in top5_indices:
                row.append(LABEL_LIST[int(index.item())])
            preds5.append(row)

    # Extract image paths
    with open('/home/jcdutoit/Snackathon/bev_classification/datasets/test_edited.txt') as f:
        image_paths = [line.rstrip() for line in f]

    # Write results
    print("Writing to /home/jcdutoit/Snackathon/top1.txt")
    image_index = 0
    with open('/home/jcdutoit/Snackathon/val_top1.txt', 'w') as f:
        for pred in preds1:
            f.write(image_paths[image_index] + ", " + pred + '\n')
            image_index += 1

    print("Writing to /home/jcdutoit/Snackathon/top5.txt")
    image_index = 0
    with open('/home/jcdutoit/Snackathon/val_top5.txt', 'w') as f:
        for row in preds5:
            for pred in row:
                f.write(image_paths[image_index] + ", " + pred)
            f.write('\n')
            image_index += 1



if __name__ == "__main__":

    ##########################
    # Train (no unfreezing)
    ##########################
    # train(start_frozen=False, model_unfreeze=0)


    ################################
    # Load and use trained model
    ################################
    # Load trained model
    model = torch.load('/home/jcdutoit/Snackathon/snack_model.pt')
    model.eval()

    # Initialize other values
    device = torch.device('cuda:0')
    test_dataset = PredictionDataset()
    test_loader = DataLoader(test_dataset,
                             shuffle=False,
                             num_workers=8,
                             batch_size=32)

    predict(model, test_loader, device)