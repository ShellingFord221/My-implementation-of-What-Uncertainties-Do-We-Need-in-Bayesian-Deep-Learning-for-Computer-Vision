# When modeling epistemic uncertainty and aleatoric uncertainty, we use MC dropout as well as loss attenuation to capture both model uncertainty and data uncertainty.
# We put distributions on both weights of the network and outputs of the network.


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


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


if not (os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
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



print('train data len: ', len(train_loader.dataset))
print('test data len: ', len(test_loader.dataset))


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(    # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,         # input height
                out_channels=16,       # n_filters
                kernel_size=5,         # filter size
                stride=1,              # filter movement/step
                padding=2,

            ),                         # output shape (16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),      # choose max value in 2x2 area, output shape (16, 14, 14)
            nn.Dropout(0.5)
        )
        self.conv2 = nn.Sequential(           # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),       # output shape (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),                  # output shape (32, 7, 7)
            nn.Dropout(0.5)
        )
        self.linear = nn.Linear(32 * 7 * 7, CLASS_NUM * 2)  # fully connected layer, output 10 classes  [batch_size, 10]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)             # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        logit = self.linear(x)
        mu, sigma = logit.split(CLASS_NUM, 1)
        return mu, sigma


cnn = CNN()
print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

best_acc = 0

elapsed_time = 0
start_time = time.time()

for epoch in range(EPOCH):
    cnn.train()
    for batch_idx, (train_x, train_y) in enumerate(train_loader):

        mu, sigma = cnn(train_x)

        prob_total = torch.zeros((NUM_SAMPLES, train_y.size(0), CLASS_NUM))
        for t in range(NUM_SAMPLES):
            # assume that each logit value is drawn from Gaussian distribution, therefore the whole logit vector is drawn from multi-dimensional Gaussian distribution
            epsilon = torch.randn(sigma.size())
            logit = mu + torch.mul(sigma, epsilon)
            prob_total[t] = F.softmax(logit, dim=1)

        prob_ave = torch.mean(prob_total, 0)
        loss = F.nll_loss(torch.log(prob_ave), train_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch: ', epoch, '| batch: ', batch_idx, '| train loss: %.4f' % loss.data.numpy())


    cnn.eval()
    cnn.apply(apply_dropout)
    correct = 0
    for batch_idx, (test_x, test_y) in enumerate(test_loader):
        prob_total = torch.zeros((NUM_SAMPLES, test_y.size(0), CLASS_NUM))
        sigma_total = torch.zeros((NUM_SAMPLES, test_y.size(0), CLASS_NUM))
        for t in range(NUM_SAMPLES):
            test_mu, test_sigma = cnn(test_x)
            prob_total[t] = F.softmax(test_mu, dim=1)
            sigma_total[t] = test_sigma

        prob_ave = torch.mean(prob_total, 0)
        pred_y = torch.max(prob_ave, 1)[1].data.numpy()
        correct += float((pred_y == test_y.data.numpy()).astype(int).sum())

        sigma_ave = torch.mean(sigma_total, 0)
        # Aleatoric uncertainty is measured by some function of sigma_ave.
        # Epistemic uncertainty is measured by some function of prob_ave (e.g. entropy).

    accuracy = correct / float(len(test_loader.dataset))
    print('-> Epoch: ', epoch, '| test accuracy: %.4f' % accuracy)
    if accuracy > best_acc:
        best_acc = accuracy




elapsed_time = time.time() - start_time
print('Best test accuracy is: ', best_acc)   # 0.9893
print('Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))


