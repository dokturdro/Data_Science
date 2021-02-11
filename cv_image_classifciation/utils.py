import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

plt.style.use('fivethirtyeight')
sns.set(style='whitegrid', color_codes=True)

def calc_accuracy(true, pred):
    pred = F.softmax(pred, dim=1)
    true = torch.zeros(pred.shape[0], pred.shape[1]).scatter_(1, true.unsqueeze(1), 1.)
    acc = (true.argmax(-1) == pred.argmax(-1)).float().detach().numpy()
    acc = float((100 * acc.sum()) / len(acc))
    return round(acc, 4)

def plot_loss(train_loss, val_loss):
    plt.plot(train_loss, label='train loss')
    plt.plot(val_loss, label='test loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('results/plot_loss.png')
    plt.close()
    print("Loss plot saved.")

def plot_accu(train_accuracy, val_accuracy):
    plt.plot(train_accuracy, label='train accuracy')
    plt.plot(val_accuracy, label='test accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Percent')
    plt.legend()
    plt.savefig('results/plot_accu.png')
    plt.close()
    print("Accu plot saved.")
