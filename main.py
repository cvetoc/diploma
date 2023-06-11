import torch
import yaml
from src.models import trainer
from src.data.datamodule import DataManager
from src.txt_logger import TXTLogger
from src.models.seq2seq_transformer import Seq2SeqTransformer


def main():
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'

    print("DEVICE =", DEVICE)

from train_seq2seq_transformer import train

if __name__ == "__main__":
    model, dm, train, val = train(True)
