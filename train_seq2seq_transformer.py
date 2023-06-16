import torch
import yaml
from src.models import trainer
from src.data.datamodule import DataManager
from src.txt_logger import TXTLogger
from src.models.seq2seq_transformer import Seq2SeqTransformer
from src.utils import graf

def train(prin=False, filename="progress_log.txt"):
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'

    print("DEVICE =", DEVICE)

    data_config = yaml.load(open("configs/data_config.yaml", 'r', encoding='utf-8'), Loader=yaml.Loader)
    dm = DataManager(data_config, DEVICE)
    train_dataloader_mlm, train_dataloader_shift = dm.prepare_data(path_data="data/russian_train_split.json", drop_last=False, aug=True)
    dev_dataloader_mlm, dev_dataloader_shift = dm.prepare_data(path_data="data/russian_dev_split.json", drop_last=True, aug=False)

    model_config = yaml.load(open("configs/model_config.yaml", 'r', encoding='utf-8'), Loader=yaml.Loader)

    model = Seq2SeqTransformer(device=DEVICE,
                               tokenizer=dm.tokenizer,
                               lr=model_config["learning_rate"],
                               sched_step=model_config["sched_step"],
                               sched_gamma=model_config["sched_gamma"]
                               ).to(DEVICE)

    logger = TXTLogger('training_logs', filename=filename)
    trainer_cls = trainer.Trainer(model=model, model_config=model_config, logger=logger, prin=prin)

    if model_config['try_one_batch']:
        for a, b in zip(train_dataloader_mlm, train_dataloader_shift):
            train_dataloader_mlm = [a]
            train_dataloader_shift = [b]
            dev_dataloader_mlm = [a]
            dev_dataloader_shift = [b]
            break

    trainer_cls.train((train_dataloader_mlm, train_dataloader_shift), (dev_dataloader_mlm, dev_dataloader_shift))

    return model, dm, (train_dataloader_mlm, train_dataloader_shift), (dev_dataloader_mlm, dev_dataloader_shift)

if __name__ == "__main__":

    model, dm, train, val = train(True)
    data_config = yaml.load(open("configs/data_config.yaml", 'r', encoding='utf-8'), Loader=yaml.Loader)
    graf(data_config["path_repository"] + "training_logs/progress_log.txt")

    torch.save(model.state_dict(), "save_model.pt")

    model_temp = torch.load("save_model.pt")

    print(dir(model_temp))

