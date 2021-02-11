import torch
import math

DEVICE = torch.device("cpu")

EPOCHS = 10
BATCH_SIZE = 25
LEARNING_RATE = 0.003
IMG_SIZE = 64
CONV_SIZE = math.floor((((IMG_SIZE - 2) / 2) - 2) / 2)
NUM_WORKER = 4

TRAIN_DATA_PATH = r"C:\datasets\images\flowers_train_test\train"
TEST_DATA_PATH = r"C:\datasets\images\flowers_train_test\test"