

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

cudnn.benchmark = True
plt.ion()   # interactive mode

from google.colab import drive
drive.mount('/content/drive')

!unzip "/content/drive/My Drive/sixClasses_T124_200.zip"

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.7223385278908533, 0.6683736685193903, 0.7070747222562316], [0.02458974013551035, 0.02741348422994171, 0.02665596268190595])
    ]),
    'val': transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.7223385278908533, 0.6683736685193903, 0.7070747222562316], [0.02458974013551035, 0.02741348422994171, 0.02665596268190595])
    ]),
}

BATCH_SIZE = 128

data_dir = 'sixClasses_T124_200'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net()

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net()

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):

        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase

        callbackTime = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            callbackInd = -1
            for inputs, labels in dataloaders[phase]:
                callbackInd += 1

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if callbackInd%callbackStep==0:
                    print('Epoch: {}, elapsed time = {}'.format(epoch, time.time()-callbackTime))
                    print('loss: {}, accuracy: {}'.format(running_loss/dataset_sizes[phase],
                                                           running_corrects.double()/dataset_sizes[phase]))
                    callbackTime = time.time()


            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('--- EVAL ---')
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            print('**************************')


            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

    def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

import gc
gc.collect()
torch.cuda.empty_cache()

model_ft = models.resnet18(pretrained=False)
model_ft.aux_logits=False


#num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
#model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

callbackPercent = 5 # every 10% precent show callback accuracy\loss
callbackStep = len(dataloaders['train'])//callbackPercent

import gc
gc.collect()
torch.cuda.empty_cache()

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=50)

torch.save(model_ft.state_dict(), '/content/drive/My Drive/MecosBloodNet(resnet1111118(evo_colab)86,1%_).pth')

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

import gc
gc.collect()
torch.cuda.empty_cache()

model_ft = models.resnet18(pretrained=False)
model_ft.aux_logits=False


#num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
#model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

callbackPercent = 5 # every 10% precent show callback accuracy\loss
callbackStep = len(dataloaders['train'])//callbackPercent

import gc
gc.collect()
torch.cuda.empty_cache()

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=50)

torch.save(model_ft.state_dict(), '/content/drive/My Drive/MecosBloodNet(resnet1111118(evo_colab)86,1%_).pth')

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.7223385278908533, 0.6683736685193903, 0.7070747222562316])
    std = np.array([0.02458974013551035, 0.02741348422994171, 0.02665596268190595])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

import cv2 as cv


def predictImg(model,im, targetSize=(200,200)):
    im = cv.resize(im, targetSize)
    im = im/255.0

    im=im.transpose((1,2,0))
    M = np.array([0.7223385278908533, 0.6683736685193903, 0.7070747222562316])
    S = np.array([0.02458974013551035, 0.02741348422994171, 0.02665596268190595])
    im = S*im + M

    batch = torch.tensor(im).unsqueeze(0)

    with torch.no_grad():
        print(batch.shape)
        batch = batch.to(device)
        out = model(im)
        _, pred = torch.argmax(out,1)
        print(pred)
    return pred



import time
import random

def checkPath(p):
    if p.split('.')[-1] == 'jpg':
        return True
    return False

model_ft.eval()

M = [0.7223385278908533, 0.6683736685193903, 0.7070747222562316]
S = [0.02458974013551035, 0.02741348422994171, 0.02665596268190595]

transform_norm = transforms.Compose(
    [
        transforms.Normalize(M, S)
    ]
)

errorMatrix = {
    class_names[0]:0,
    class_names[1]:0,
    class_names[2]:0,
    class_names[3]:0,
    class_names[4]:0,
    class_names[5]:0
}


manualEvalDir = r'sixClasses_T124_200/val/neut'
startTime = time.time()


iterDir = os.listdir(manualEvalDir)
random.shuffle( iterDir )

for i,imPath in enumerate( iterDir ):
    imPath = os.path.join(manualEvalDir,imPath)

    if not checkPath(imPath): continue;
    #print(imPath)

    with torch.no_grad():
        img = torchvision.io.read_image(imPath)/255

        img_show = img.numpy().transpose((1, 2, 0))
        #plt.imshow(img_show)

        img = transform_norm(img)

        img = img.unsqueeze(0)
        res = model_ft(img.to(device))
        res = torch.argmax(res, 1)

        #print(class_names[res])

        errorMatrix[class_names[res]] += 1


        if i%250==0:
            print('{} of {}'.format(i,27912))
            if i>10000:
                print('Elapsed time = {}'.format(time.time() - startTime))
                print(errorMatrix)

errorMatrix

from torchvision.models.regnet import BlockParams
basco {'baso': 200, 'blas': 17, 'eosi': 14, 'lymph': 33, 'mono': 22, 'neut': 87}
Blas  {'baso': 0, 'blas': 543, 'eosi': 0, 'lymph': 66, 'mono': 10, 'neut': 2}
eosi  {'baso': 5, 'blas': 1, 'eosi': 831, 'lymph': 10, 'mono': 6, 'neut': 79}
lymph {'baso': 0, 'blas': 49, 'eosi': 1, 'lymph': 532, 'mono': 1, 'neut': 2}
mono  {'baso': 4, 'blas': 27, 'eosi': 0, 'lymph': 103, 'mono': 314, 'neut': 3}
neut  {'baso': 3, 'blas': 4, 'eosi': 3, 'lymph': 43, 'mono': 10, 'neut': 668}





