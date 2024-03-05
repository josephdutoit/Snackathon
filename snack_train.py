from torchvision.models import resnet152
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torch import optim, nn
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import gc
import yaml

# Get target labels and create a dictionary of them
with open('bev_classification/names.yaml', 'r') as f:
    LABEL_LIST = yaml.full_load(f)['classes']
LABEL_DICT = dict([(LABEL_LIST[i], i) for i in range(len(LABEL_LIST))])

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, train=False):
        self.data_dir = '/home/jcdutoit/Snackathon/bev_classification'

        # Get training data
        if train:
          self.data_labels = "/home/jcdutoit/Snackathon/bev_classification/datasets/train.txt"
          raw_txt = np.loadtxt(self.data_labels, dtype=str)
          lines_labels = np.array([line.split(',') for line in raw_txt])

        # Get testing data
        else:
          self.data_labels = "/home/jcdutoit/Snackathon/bev_classification/datasets/test_edited.txt"
          raw_txt = np.loadtxt(self.data_labels, dtype=str)
          lines_labels = np.array([[line, line.split('/')[2]] for line in raw_txt])

        # Store the image paths and labels in a dataframe
        self.df = pd.DataFrame(lines_labels, columns=['image', 'label'])
        self.transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.PILToTensor()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the image and label from the file
        img_path = os.path.join(self.data_dir, self.df.iloc[idx, 0])
        image = Image.open(img_path)
        label = LABEL_DICT[self.df.iloc[idx, 1]]
        l_tensor = torch.tensor(label)

        # Transform the image
        image_t = self.transform(image).float()
        return image_t, l_tensor


class CustomResNet(nn.Module):
    def __init__(self, num_classes, start_frozen=False):
        super(CustomResNet, self).__init__()

        self.res_model = resnet152(pretrained=True)

        # For frozen start
        if start_frozen:
            for param in self.res_model.parameters():
                param.requires_grad = False
        
        # Last layer of the resnet
        self.res_model.fc = nn.Linear(self.res_model.fc.in_features, 99)

    def unfreeze(self, n_layers):
        # For unfreezing. Didn't work as well for us
        child_list = list(self.res_model.children())
        for child in child_list[len(child_list) - n_layers - 1:]:
           for param in child.parameters():
               param.requires_grad = True

    def forward(self, x):
        return self.res_model(x)

def accuracy(y_hat, y_truth):
    # Basic accuracy function
    preds = torch.argmax(y_hat, dim=1)
    acc = torch.mean((preds == y_truth).float())
    return acc

def evaluate(model, objective, val_loader, device, epoch):
    val_losses = 0
    val_accs = 0
    batches = 0
    model.eval()
    preds = []
    top5_preds = []
    with torch.no_grad():
        for x, y_truth in val_loader:

            batches += 1

            # Get validation loss and predictions
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

    # Write the top1 and top5 predictions to file
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

    # Initialize datasets and dataloaders
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

    # Main training loop
    cnt = 0
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        if cnt < model_unfreeze and cnt % 45 == 0:
            # Try unfreezing. It didn't work so well for us.
            layers = int(cnt / 45)
            print("\nUnfreezing " + str(layers) + " layers")
            model.unfreeze(layers)

        for x, y_truth in train_loader:
            
            x, y_truth = x.to(device), y_truth.to(device)

            optimizer.zero_grad()

            # Training
            y_hat = model(x)
            train_loss = objective(y_hat, y_truth)
            train_acc = accuracy(y_hat, y_truth)

            train_loss.backward()
            optimizer.step()

            train_accs.append(train_acc)
            train_losses.append(train_loss.item())

            # Validation
            if cnt % val_every == 0:
                val_loss, val_acc = evaluate(model, objective, val_loader, device, cnt)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                torch.save(model, '/home/jcdutoit/Snackathon/' + str(epoch) + '_snack_model.pt')

            # Update progress bar
            pbar.set_description('train loss:{:.4f}, train accuracy:{:.4f}, val loss:{:.4f}, val accuracy:{:.4f}.'.format(train_loss.item(), train_acc, val_losses[-1], val_accs[-1]))
            pbar.update(1)
            cnt += 1

    pbar.close()

if __name__ == "__main__":
    # Train with no unfreezing
    train(start_frozen=False, model_unfreeze=0)

