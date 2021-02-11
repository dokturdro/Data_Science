import numpy as np
import time
import torch.nn as nn

from model import ensemble_Net
from data_loader import data_loader
from utils import *
from global_params import *

if __name__ == '__main__':

    train_data_loader, test_data_loader = data_loader(TRAIN_DATA_PATH, TEST_DATA_PATH)
    model = ensemble_Net()
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=LEARNING_RATE)

    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []

    for epoch in range(EPOCHS):

        start = time.time()

        train_epoch_loss = []
        train_epoch_accuracy = []
        val_epoch_loss = []
        val_epoch_accuracy = []

        for images, labels in train_data_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            preds = model(images)

            acc = calc_accuracy(labels.cpu(), preds.cpu())
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            train_epoch_loss.append(loss_value)
            train_epoch_accuracy.append(acc)

        for images, labels in test_data_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            preds = model(images)
            acc = calc_accuracy(labels.cpu(), preds.cpu())
            loss = criterion(preds, labels)

            loss_value = loss.item()
            val_epoch_loss.append(loss_value)
            val_epoch_accuracy.append(acc)

        train_epoch_loss = np.mean(train_epoch_loss)
        train_epoch_accuracy = np.mean(train_epoch_accuracy)
        val_epoch_loss = np.mean(val_epoch_loss)
        val_epoch_accuracy = np.mean(val_epoch_accuracy)

        end = time.time()

        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)
        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)

        print("@@ Epoch {} = {}s".format(epoch, int(end - start)))
        print("Train Loss = {}".format(round(train_epoch_loss, 3)))
        print("Train Accu = {} %".format(train_epoch_accuracy))
        print("Valid Loss = {}".format(round(val_epoch_loss, 3)))
        print("Valid Accu = {} % \n".format(val_epoch_accuracy))

    plot_loss(train_loss, val_loss)
    plot_accu(train_accuracy, val_accuracy)

    exit()




