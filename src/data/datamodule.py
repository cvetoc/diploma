from torch.utils.data import DataLoader
import json
import os
from random import randint

from tqdm import tqdm

from src.data.mt_dataset import MTDataset_mlm, MTDataset_shift
from src.data.tokenizers.unif_tokenizers import UNIFTokenizer
from src.data.mm_augmentation import MM_Augmentation
from src.data.augmetator import Augmetator_func_tool


class DataManager:
    def __init__(self, config, device):
        super().__init__()
        self.tokenizer = None
        self.source_tokenizer = None
        self.target_tokenizer = None
        self.config = config
        self.device = device

        self.tokenizer = UNIFTokenizer(path=self.config["path_repository"] + "data/query_vocab.json", pad_flag=True,
                                       max_length=self.config["max_sent_len"])

    def prepare_data(self, path_data, drop_last=False, aug=False):

        dev_data = json.load(open(os.path.join(self.config["path_repository"] + path_data), 'r', encoding="utf-8"))
        target_sentences = []
        source_sentences = []
        for sample in tqdm(dev_data[:self.config["separate_batch"]], desc="Pars data"):
            target_sentences.append(sample['masked_sparql'])
            source_sentences.append(sample['question'])

        if aug:
            aug_index = int(0.3*len(target_sentences))
            aug_target_sentences = target_sentences[:aug_index]
            aug_source_sentences = source_sentences[:aug_index]

            tools = Augmetator_func_tool()
            augmentor = MM_Augmentation(tools)
            aug_source_sentences = augmentor.run(aug_source_sentences)

            target_sentences = target_sentences + aug_target_sentences
            source_sentences = source_sentences + aug_source_sentences

        source_sentences_true = source_sentences.copy()

        target_sentences_mlm = target_sentences.copy()
        source_sentences_mlm = source_sentences.copy()

        target_sentences_mlm_temp = []
        for sentence in tqdm(target_sentences_mlm, desc="data mlm"):
            temp_list = sentence.split()
            kol_mask = int(len(temp_list) * 0.15)
            for word_i in range(len(temp_list)):
                if ':' in temp_list[word_i]:
                    temp_list[word_i] = self.tokenizer.tokenizer.mask_token
                    kol_mask -= 1
                if kol_mask == 0:
                    break
            target_sentences_mlm_temp.append(' '.join(temp_list))
        target_sentences_mlm = target_sentences_mlm_temp.copy()
        del target_sentences_mlm_temp

        target_sentences_shift = target_sentences.copy()
        source_sentences_shift = source_sentences.copy()

        sep_border = int(len(target_sentences_shift) * 0.5)

        sep = randint(1, sep_border)
        source_sentences_shift = source_sentences_shift[sep:sep_border] + source_sentences_shift[
                                                                          :sep] + source_sentences_shift[sep_border:]

        class_label = [[0] for _ in range(sep_border)]
        for _ in range(len(source_sentences_shift) - sep_border):
            class_label.append([1])

        tokenized_source_sentences_mlm = [self.tokenizer(i, j) for i, j in
                                          zip(source_sentences_mlm, target_sentences_mlm)]
        tokenized_target_sentences_mlm = [self.tokenizer(i, j) for i, j in zip(source_sentences_true, target_sentences)]
        tokenized_source_sentences_shift = [self.tokenizer(i, j) for i, j in
                                            zip(source_sentences_shift, target_sentences_shift)]

        dataset_mlm = MTDataset_mlm(tokenized_source_list=tokenized_source_sentences_mlm,
                                      tokenized_target_list=tokenized_target_sentences_mlm, dev=self.device)

        dataset_shift = MTDataset_shift(tokenized_source_list=tokenized_source_sentences_shift,
                                      tokenized_target_list=class_label, dev=self.device)

        dataloader_mlm = DataLoader(dataset_mlm, shuffle=True,
                                    batch_size=self.config["batch_size"], drop_last=drop_last)

        dataloader_shift = DataLoader(dataset_shift, shuffle=True,
                                      batch_size=self.config["batch_size"], drop_last=drop_last)


        return dataloader_mlm, dataloader_shift

import yaml

if __name__ == "__main__":

    path = "D:/ФизТех/Учеба в маге/2 курс (10сем)/Кафедральные/Машинное обучение продвинутый уровень/статья/Project_cod/"
    data_config = yaml.load(open(path + "configs/data_config.yaml", 'r'), Loader=yaml.Loader)
    data_config["path_repository"] = path
    dm = DataManager(data_config, "cpu")

    dataloader_mlm, dataloader_shift = dm.prepare_data(path_data="data/russian_dev_split.json", drop_last=False, aug=True)

    for batch in dataloader_mlm:
        print(batch[0][0], batch[1][0], batch[2][0], batch[3][0])
        break

    for batch in dataloader_shift:
        print(batch[0][0], batch[1][0], batch[2][0], batch[3][0])
        break

    print("OK")
