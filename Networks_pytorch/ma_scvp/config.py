import torch

LEARNING_RATE = 1e-4 
NUM_EPOCHS = 301
BATCH_SIZE = 8
NUM_WORKERS = 4
LOSS_LAMBDA = 1.5
LOSS = "NBVLoss"
THRESHOLD_GAMMA = 0.5
VALIDATION_SPILT = 0.8

NET_TYPE = "SCVP"

SHUFFLE_DATASET = True
RANDOM_SEED = 42
GRID_PATH = "/home/huhao/code_python/Datasets/LongTail_MA-SCVP/MASCVP_NBVSample_8_grid.npy"
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
OPTIMIZER = "Adam"

LOAD_MODEL = False
LOAD_PATH = "nbvsample32_lambda3.5_bsize8_lr0.0006.pth.tar"
SAVE_MODEL = True
SAVE_PATH = "scvp_nbvsample8_lambda1.5_bsize8_lr0.0004.pth.tar"

