import torch
from pathlib import Path

# Paths
BASE_PATH = Path(__file__).resolve().parents[1]
TRAIN_VYANJAN_PATH = BASE_PATH / "data" / "Train_vyanjan"
TEST_VYANJAN_PATH = BASE_PATH / "data" / "Test_vyanjan"
TRAIN_DIGIT_PATH = BASE_PATH / "data" / "Train_digits"
TEST_DIGIT_PATH = BASE_PATH / "data" / "Test_digits"
BEST_MODEL_VYANJAN = BASE_PATH / "models" / "best_vyanjan_model.pt"
BEST_MODEL_DIGIT = BASE_PATH / "models" / "best_digit_model.pt"
BEST_MODEL_PATH = ""
INDEX_DIGIT = BASE_PATH / "src" / "index_to_digit.json"
INDEX_VYNAJAN = BASE_PATH / "src" / "index_to_vyanjan.json"


# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-5

# Miscellanous
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
INTERVAL = 100
