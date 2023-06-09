from transformers import AutoTokenizer

import json


class UNIFTokenizer:
    def __init__(self, path="data/query_vocab.json", pad_flag=False, max_length=100):

        self.tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
        token_list = json.load(open(path, 'r'))

        token_list = set(token_list) - set(self.tokenizer.vocab.keys())
        self.tokenizer.add_tokens(list(token_list))

        self.max_sent_len = max_length
        self.pad_flag = pad_flag

    def __call__(self, text_list_ru, text_list_ques):

        if self.pad_flag:
            token_list = self.tokenizer(text_list_ru, text_list_ques,
                                        max_length=self.max_sent_len,
                                        padding='max_length',
                                        truncation=True,
                                        return_tensors="pt")
        else:
            token_list = self.tokenizer(text_list_ru, text_list_ques,
                                        return_tensors="pt")

        return token_list

    def __len__(self):
        return len(self.tokenizer)

    def decode(self, token_list):
        # TODO IRM
        token_list = self.tokenizer.decode(token_list)
        # декодер удаляет пробелы перед спецсимволами
        token_list = ' ?'.join(token_list.split('?'))
        return token_list

import os
import yaml

from tqdm import tqdm
from random import shuffle

if __name__ == "__main__":

    path = "D:/ФизТех/Учеба в маге/2 курс (10сем)/Кафедральные/Машинное обучение продвинутый уровень/статья/Project_cod/"
    data_config = yaml.load(open(path + "configs/data_config.yaml", 'r'), Loader=yaml.Loader)
    dev_data = json.load(open(os.path.join(path + "data/russian_dev_split.json"), 'r', encoding="utf-8"))
    shuffle(dev_data)
    target_sentences = []
    source_sentences = []
    for sample in tqdm(dev_data[:3], desc="Pars data"):
        target_sentences.append(sample['masked_sparql'])
        source_sentences.append(sample['question'])

    tokenizer = UNIFTokenizer(path=path + "data/query_vocab.json", pad_flag=True,
                              max_length=data_config["max_sent_len"])

    print([(i, j) for i, j in zip(source_sentences, target_sentences)])
    tok_list = [tokenizer(i, j)['input_ids'][0] for i, j in zip(source_sentences, target_sentences)]
    print(tok_list)
    print([tokenizer.decode(i) for i in tok_list])


