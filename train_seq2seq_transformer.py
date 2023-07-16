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
    data_path = lambda x: data_config["path_repository"] + "data/" + data_config["data_language"] + str(x) + data_config["data_name_file"] + ".json"
    dm = DataManager(data_config, DEVICE)
    train_dataloader_mlm, train_dataloader_shift = dm.prepare_data(path_data=data_path("train"), drop_last=False, aug=0.3)
    dev_dataloader_mlm, dev_dataloader_shift = dm.prepare_data(path_data=data_path("dev"), drop_last=True)

    model_config = yaml.load(open(data_config["path_repository"] + "configs/model_config.yaml", 'r', encoding='utf-8'), Loader=yaml.Loader)

    model = Seq2SeqTransformer(device=DEVICE,
                               tokenizer=dm.tokenizer,
                               model_path=model_config["pre_train_model"],
                               lr=model_config["learning_rate"],
                               sched_step=model_config["sched_step"],
                               sched_gamma=model_config["sched_gamma"]
                               ).to(DEVICE)

    logger = TXTLogger(data_config["path_repository"] + 'training_logs', filename=filename)
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

    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    torch.save(model.state_dict(), data_config["path_repository"] + "save_model.pt")

