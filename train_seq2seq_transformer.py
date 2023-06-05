import torch
import yaml
from src.models import trainer
from src.data.datamodule import DataManager
from src.txt_logger import TXTLogger
from src.models.seq2seq_transformer import Seq2SeqTransformer

def pre_train(prin=False, filename="aug_progress_log.txt"):
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'

    print("DEVICE =", DEVICE)

    data_config = yaml.load(open("configs/data_config.yaml", 'r'), Loader=yaml.Loader)
    dm = DataManager(data_config, DEVICE)
    train_dataloader = dm.prepare_data(path_data="data/russian_train_split.json", drop_last=False, aug=True)
    dev_dataloader = dm.prepare_data(path_data="data/russian_dev_split.json", drop_last=True, aug=False)

    model_config = yaml.load(open("configs/model_config.yaml", 'r'), Loader=yaml.Loader)

    model = Seq2SeqTransformer(device=DEVICE,
                               tokenizer=dm.tokenizer,
                               lr=model_config["learning_rate"]
                               ).to(DEVICE)

    logger = TXTLogger('training_logs', filename='pre_'+filename)
    trainer_cls = trainer.Trainer(model=model, model_config=model_config, logger=logger, prin=prin)

    # print(list(train_dataloader))
    # if model_config['try_one_batch']:
    #     train_dataloader = [list(train_dataloader)[0]]
    #     dev_dataloader = [list(train_dataloader)[0]]

    trainer_cls.train(train_dataloader, dev_dataloader)

    train_dataloader = dm.prepare_data(path_data="data/russian_dev_split.json", drop_last=False, aug=False)
    dev_dataloader = dm.prepare_data(path_data="data/russian_test_split.json", drop_last=True, aug=False)

    model.lr = 0.0001
    model.sched_step = 1000
    model.reset_learn()

    logger = TXTLogger('training_logs', filename=filename)
    trainer_cls = trainer.Trainer(model=model, model_config=model_config, logger=logger, prin=prin)
    trainer_cls.train(train_dataloader, dev_dataloader)

    return model, dm, train_dataloader, dev_dataloader
def train(prin=False, filename="progress_log.txt"):
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'

    print("DEVICE =", DEVICE)

    data_config = yaml.load(open("configs/data_config.yaml", 'r'), Loader=yaml.Loader)
    dm = DataManager(data_config, DEVICE)
    train_dataloader = dm.prepare_data(path_data="data/russian_dev_split.json", drop_last=False, aug=False)
    dev_dataloader = dm.prepare_data(path_data="data/russian_test_split.json", drop_last=True, aug=False)

    model_config = yaml.load(open("configs/model_config.yaml", 'r'), Loader=yaml.Loader)

    model = Seq2SeqTransformer(device=DEVICE,
                               tokenizer=dm.tokenizer,
                               lr=model_config["learning_rate"]
                               ).to(DEVICE)

    logger = TXTLogger('training_logs', filename=filename)
    trainer_cls = trainer.Trainer(model=model, model_config=model_config, logger=logger, prin=prin)

    # print(list(train_dataloader))
    # if model_config['try_one_batch']:
    #     train_dataloader = [list(train_dataloader)[0]]
    #     dev_dataloader = [list(train_dataloader)[0]]

    trainer_cls.train(train_dataloader, dev_dataloader)

    return model, dm, train_dataloader, dev_dataloader


if __name__ == "__main__":
    model, dm, train, val = train(True)