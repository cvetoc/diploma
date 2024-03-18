import torch
import yaml
from src.models import trainer
from src.data.datamodule import DataManager_pretrain, DataManager
from src.txt_logger import TXTLogger
#from src.models.seq2seq_transformer import Seq2SeqTransformer
from src.models.bert_model import Seq2SeqTransformer
from src.utils import pre_graf, graf

def train(prin=False, filename="progress_log_train.txt", model_state_path=None, freez=False):
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'

    print("DEVICE =", DEVICE)

    data_config = yaml.load(open("configs/data_config.yaml", 'r', encoding='utf-8'), Loader=yaml.Loader)
    data_path = lambda x: data_config["path_repository"] + "data/" + data_config["data_language"] + str(x) + data_config["data_name_file"] + ".json"

    model_config = yaml.load(open(data_config["path_repository"] + "configs/model_config.yaml", 'r', encoding='utf-8'), Loader=yaml.Loader)

    dm_ = DataManager(data_config, DEVICE)
    dev_dataloader = dm_.prepare_data(path_data=data_path("dev"), drop_last=False)
    test_dataloader = dm_.prepare_data(path_data=data_path("test"), drop_last=True)

    model = Seq2SeqTransformer(device=DEVICE,
                               tokenizer=dm_.tokenizer,
                               model_path=model_config["pre_train_model"],
                               lr=model_config["learning_rate"],
                               sched_step=model_config["sched_step"],
                               sched_gamma=model_config["sched_gamma"]
                               ).to(DEVICE)

    if model_state_path:
        model.load_state_dict(torch.load(model_state_path, map_location=torch.device(DEVICE)))

    if freez:
        for i, param in enumerate(model.parameters()):
            if i not in [199, 200]:
                param.requires_grad = True
            else:
                param.requires_grad = False

    logger_ = TXTLogger(data_config["path_repository"] + 'training_logs_', filename=filename)
    trainer_cls_ = trainer.Trainer(model=model, model_config=model_config, logger=logger_, prin=prin)

    if model_config['try_one_batch']:
        for a in dev_dataloader:
            dev_dataloader = [a]
            test_dataloader = [a]
            break

    trainer_cls_.train(dev_dataloader, test_dataloader)

    return model, dm_, (dev_dataloader, test_dataloader)


def pre_train(prin=False, filename="progress_log_pre_train.txt"):
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'

    print("DEVICE =", DEVICE)

    data_config = yaml.load(open("configs/data_config.yaml", 'r', encoding='utf-8'), Loader=yaml.Loader)
    data_path = lambda x: data_config["path_repository"] + "data/" + data_config["data_language"] + str(x) + data_config["data_name_file"] + ".json"
    dm = DataManager_pretrain(data_config, DEVICE)
    train_dataloader_mlm, train_dataloader_shift = dm.prepare_data(path_data=data_path("train"), drop_last=False, aug=0.3)
    dev_dataloader_mlm, dev_dataloader_shift = dm.prepare_data(path_data=data_path("dev"), drop_last=True)

    model_config = yaml.load(open(data_config["path_repository"] + "configs/pre_model_config.yaml", 'r', encoding='utf-8'), Loader=yaml.Loader)

    model = Seq2SeqTransformer(device=DEVICE,
                               tokenizer=dm.tokenizer,
                               model_path=model_config["pre_train_model"],
                               lr=model_config["learning_rate"],
                               sched_step=model_config["sched_step"],
                               sched_gamma=model_config["sched_gamma"]
                               ).to(DEVICE)

    logger = TXTLogger(data_config["path_repository"] + 'pre_training_logs', filename=filename)
    trainer_cls = trainer.Trainer(model=model, model_config=model_config, logger=logger, prin=prin)

    if model_config['try_one_batch']:
        for a, b in zip(train_dataloader_mlm, train_dataloader_shift):
            train_dataloader_mlm = [a]
            train_dataloader_shift = [b]
            dev_dataloader_mlm = [a]
            dev_dataloader_shift = [b]
            break

    trainer_cls.pre_train((train_dataloader_mlm, train_dataloader_shift), (dev_dataloader_mlm, dev_dataloader_shift))

    return model, dm, ((train_dataloader_mlm, train_dataloader_shift), (dev_dataloader_mlm, dev_dataloader_shift))


def pre_train_train(prin=False, filename="progress_log.txt", callback=None):
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'

    print("DEVICE =", DEVICE)

    data_config = yaml.load(open("configs/data_config.yaml", 'r', encoding='utf-8'), Loader=yaml.Loader)
    data_path = lambda x: data_config["path_repository"] + "data/" + data_config["data_language"] + str(x) + data_config["data_name_file"] + ".json"
    dm = DataManager_pretrain(data_config, DEVICE)
    train_dataloader_mlm, train_dataloader_shift = dm.prepare_data(path_data=data_path("train"), drop_last=False, aug=0.3)
    dev_dataloader_mlm, dev_dataloader_shift = dm.prepare_data(path_data=data_path("dev"), drop_last=True)

    model_config = yaml.load(open(data_config["path_repository"] + "configs/pre_model_config.yaml", 'r', encoding='utf-8'), Loader=yaml.Loader)

    model = Seq2SeqTransformer(device=DEVICE,
                               tokenizer=dm.tokenizer,
                               model_path=model_config["pre_train_model"],
                               lr=model_config["learning_rate"],
                               sched_step=model_config["sched_step"],
                               sched_gamma=model_config["sched_gamma"]
                               ).to(DEVICE)

    logger = TXTLogger(data_config["path_repository"] + 'pre_training_logs', filename=filename)
    trainer_cls = trainer.Trainer(model=model, model_config=model_config, logger=logger, prin=prin)

    if model_config['try_one_batch']:
        for a, b in zip(train_dataloader_mlm, train_dataloader_shift):
            train_dataloader_mlm = [a]
            train_dataloader_shift = [b]
            dev_dataloader_mlm = [a]
            dev_dataloader_shift = [b]
            break

    trainer_cls.pre_train((train_dataloader_mlm, train_dataloader_shift), (dev_dataloader_mlm, dev_dataloader_shift))

    # ------------------------------------------------------------------------------------------------------------------
    if callback:
        callback(model)
    # ------------------------------------------------------------------------------------------------------------------

    dm_ = DataManager(data_config, DEVICE)
    dev_dataloader = dm_.prepare_data(path_data=data_path("dev"), drop_last=False)
    test_dataloader = dm_.prepare_data(path_data=data_path("test"), drop_last=True)

    model_config_ = yaml.load(open(data_config["path_repository"] + "configs/model_config.yaml", 'r', encoding='utf-8'),
                             Loader=yaml.Loader)


    logger_ = TXTLogger(data_config["path_repository"] + 'training_logs', filename=filename)
    trainer_cls_ = trainer.Trainer(model=model, model_config=model_config_, logger=logger_, prin=prin)

    if model_config['try_one_batch']:
        for a, b in zip(dev_dataloader, test_dataloader):
            dev_dataloader = [a]
            test_dataloader = [a]
            break

    trainer_cls_.train(dev_dataloader, test_dataloader)

    return model, dm, ((train_dataloader_mlm, train_dataloader_shift), (dev_dataloader_mlm, dev_dataloader_shift))

if __name__ == "__main__":
    # TODO: дописать для двойной модели

    # model, dm, train, val = pre_train(True)
    # data_config = yaml.load(open("configs/data_config.yaml", 'r', encoding='utf-8'), Loader=yaml.Loader)
    # pre_graf(data_config["path_repository"] + "training_logs/progress_log.txt")
    # graf(data_config["path_repository"] + "training_logs_/progress_log.txt")

    model, dm, data = train(True, freez=True)
    data_config = yaml.load(open("configs/data_config.yaml", 'r', encoding='utf-8'), Loader=yaml.Loader)
    graf(data_config["path_repository"] + "training_logs_/progress_log_train.txt")

    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    torch.save(model.state_dict(), data_config["path_repository"] + "save_model.pt")

