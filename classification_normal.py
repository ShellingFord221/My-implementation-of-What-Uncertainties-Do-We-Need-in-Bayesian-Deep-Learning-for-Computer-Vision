# This is a normal neural network.


import os
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision import datasets, transforms
import torch.nn.functional as F



EPOCH = 10
BATCH_SIZE = 100
LR = 0.001
DOWNLOAD_MNIST = False
CLASS_NUM = 10
NUM_SAMPLES = 10


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s



if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    DOWNLOAD_MNIST = True


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist', train=True, download=DOWNLOAD_MNIST,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./mnist', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
            nn.Dropout(0.5)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
            nn.Dropout(0.5)
        )
        self.linear = nn.Linear(32 * 7 * 7, CLASS_NUM)  # fully connected layer, output 10 classes  [batch_size, 10]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        logit = self.linear(x)
        mu = logit
        return mu


cnn = CNN()
print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()


best_acc = 0

elapsed_time = 0
start_time = time.time()

for epoch in range(EPOCH):
    cnn.train()
    for batch_idx, (train_x, train_y) in enumerate(train_loader):

        mu = cnn(train_x)
        loss = loss_func(mu, train_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch: ', epoch, '| batch: ', batch_idx, '| train loss: %.4f' % loss.data.numpy())


    cnn.eval()
    correct = 0
    for batch_idx, (test_x, test_y) in enumerate(test_loader):

        test_mu = cnn(test_x)
        prob = F.softmax(test_mu, dim=1)

        pred_y = torch.max(prob, 1)[1].data.numpy()
        correct += float((pred_y == test_y.data.numpy()).astype(int).sum())


    accuracy = correct / float(len(test_loader.dataset))
    print('-> Epoch: ', epoch, '| test accuracy: %.4f' % accuracy)
    if accuracy > best_acc:
        best_acc = accuracy





elapsed_time = time.time() - start_time
print('Best test accuracy is: ', best_acc)   # 0.9914
print('Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))

