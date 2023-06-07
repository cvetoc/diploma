"""
import json
import os

from tqdm import tqdm

from src.data.tokenizers.query_space_tokenizer import QuerySpaceTokenizer
from src.data.tokenizers.ru_tokenizer import RUTokenizer

dev_data = json.load(open(os.path.join("data/russian_test_split.json"), 'r', encoding="utf-8"))
dev_sparql_list = [sample['masked_sparql'] for sample in dev_data]
dev_questions_list = [sample['question'] for sample in dev_data]


# tokenizer = QuerySpaceTokenizer(dev_sparql_list, "data/query_vocab.json", pad_flag=True)
# token = tokenizer(dev_sparql_list[0])
# print(token)
# print(tokenizer.decode(token))
# print(tokenizer.word2index)

tokenizer = RUTokenizer(False, 10000)
# max_len = -1
# for i in tqdm(dev_questions_list):
#     len_ = len(tokenizer(i))
#     if max_len < len_:
#         max_len = len_
#
# print(max_len)

print(tokenizer.tokenizer.vocab)

# print(dev_questions_list[0])
# print(tokenizer.tokenizer(dev_questions_list[0], return_tensors="pt"))
#
# token = tokenizer(dev_questions_list[0])
# print(token)
# print(tokenizer.decode(token))
"""
"""
import json
import os

from src.data.mm_augmentation import MM_Augmentation
from src.data.augmetator import Augmetator_func_tool

dev_data = json.load(open(os.path.join("data/russian_test_split.json"), 'r', encoding="utf-8"))
dev_sparql_list = [sample['masked_sparql'] for sample in dev_data]
dev_questions_list = [sample['question'] for sample in dev_data]

tools = Augmetator_func_tool()
augmentor = MM_Augmentation(tools)
print(augmentor.run(dev_questions_list))
"""
"""
import yaml
import torch

from src.data.datamodule import DataManager

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = 'cpu'

print("DEVICE =", DEVICE)

data_config = yaml.load(open("configs/data_config.yaml", 'r'), Loader=yaml.Loader)
dm = DataManager(data_config, DEVICE)

dev_dataloader = dm.prepare_data(path_data="data/russian_dev_split.json", drop_last=False, aug=True)

for ex in dev_dataloader:
    ex_0 = ex
    break

for i, j in zip(*ex_0):
    print("-------------------------------------------------------------------")
    print(dm.tokenizer.decode(i.tolist()))
    print(j[0], dm.tokenizer.decode(j[1:].tolist()))
"""
import torch
import yaml
from src.models import trainer
from src.data.datamodule import DataManager
from src.txt_logger import TXTLogger
from src.models.seq2seq_transformer import Seq2SeqTransformer


def main(prin=False):
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'

    print("DEVICE =", DEVICE)

    data_config = yaml.load(open("configs/data_config.yaml", 'r'), Loader=yaml.Loader)
    dm = DataManager(data_config, DEVICE)
    train_dataloader = dm.prepare_data(path_data="data/russian_dev_split.json", drop_last=False, aug=True)
    dev_dataloader = dm.prepare_data(path_data="data/russian_test_split.json", drop_last=True, aug=False)

    model_config = yaml.load(open("configs/model_config.yaml", 'r'), Loader=yaml.Loader)

    model = Seq2SeqTransformer(device=DEVICE,
                               tokenizer=dm.tokenizer,
                               lr=model_config["learning_rate"]
                               ).to(DEVICE)

    logger = TXTLogger('training_logs')
    trainer_cls = trainer.Trainer(model=model, model_config=model_config, logger=logger, prin=prin)

    # print(list(train_dataloader))
    # if model_config['try_one_batch']:
    #     train_dataloader = [list(train_dataloader)[0]]
    #     dev_dataloader = [list(train_dataloader)[0]]

    trainer_cls.train(train_dataloader, dev_dataloader)

    return model, dm, train_dataloader, dev_dataloader

from train_seq2seq_transformer import pre_train

if __name__ == "__main__":
    #model, dm, train, val = main(True)
    model, dm, train, val = pre_train(True)
