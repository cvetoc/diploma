from torch.utils.data import DataLoader
import json
import os
from random import randint
import numpy as np
import re
import copy

from tqdm import tqdm

from src.data.mt_dataset import MTDataset_mlm, MTDataset_shift, MTDataset
from src.data.tokenizers.unif_tokenizers import UNIFTokenizer
from src.data.mm_augmentation import MM_Augmentation
from src.data.augmetator import Augmetator_func_tool


class DataManager_pretrain:
    def __init__(self, config, device):
        super().__init__()
        self.tokenizer = None
        self.config = config
        self.device = device

        self.db2attr_dict = json.load(open(os.path.join(self.config["path_repository"],
                                                        self.config["path_data_schema"]), 'r', encoding="utf8"))

        self.tokenizer = UNIFTokenizer(path_tok=self.config["path_repository"] + self.config["path_query_vocab"], 
                                       pre_train_name=self.config["pre_train_tokenizer"],
                                       pad_flag=True,
                                       max_length=self.config["max_sent_len"])

    def _aug_call(self, target, sorce, aug=0.3):
        aug_index = int(aug * len(target))
        aug_target = target[:aug_index]
        aug_source = sorce[:aug_index]

        tools = Augmetator_func_tool()
        augmentor = MM_Augmentation(tools)
        aug_source = augmentor.run(aug_source)

        return aug_target, aug_source

    def prepare_data(self, path_data, drop_last=False, aug=None):

        dev_data = json.load(open(os.path.join(path_data), 'r', encoding="utf-8"))
        target_sentences = []
        source_sentences = []
        kb_id_sentences = []
        for sample in tqdm(dev_data[:self.config["separate_batch"]], desc="Pars data"):
            target_sentences.append(sample['masked_query'])
            source_sentences.append(sample['question'])
            kb_id_sentences.append(sample['kb_id'])

        sep_flag = int(len(target_sentences) / 2)

        target_sentences_mlm = target_sentences[:sep_flag].copy()
        source_kb_id_mlm = kb_id_sentences[:sep_flag].copy()
        source_sentences_mlm = source_sentences[:sep_flag].copy()

        target_sentences_shift = target_sentences[sep_flag:2 * sep_flag].copy()
        source_kb_id_shift = kb_id_sentences[sep_flag:2 * sep_flag].copy()
        source_sentences_shift = source_sentences[sep_flag:2 * sep_flag].copy()

#         if aug is not None:
#             aug_target_sentences, aug_source_sentences = self._aug_call(target_sentences_mlm, source_sentences_mlm, aug)

#             target_sentences_mlm = target_sentences_mlm + aug_target_sentences
#             source_sentences_mlm = source_sentences_mlm + aug_source_sentences

#             aug_target_sentences, aug_source_sentences = self._aug_call(target_sentences_shift, source_sentences_shift,
#                                                                         aug)

#             target_sentences_shift = target_sentences_shift + aug_target_sentences
#             source_sentences_shift = source_sentences_shift + aug_source_sentences

        # MLM_last

#         source_sentences_true = source_sentences_mlm.copy()
#         target_sentences_mlm_temp = []
#         for sentence in tqdm(target_sentences_mlm, desc="data mlm"):
#             temp_list = sentence.split()
#             index_temp_list = set(range(len(temp_list)))
#             for i, word in enumerate(temp_list):
#                 if not re.search('[a-zA-Z0-9]', word):
#                     index_temp_list = index_temp_list - set([i])

#             index_masc = np.random.choice(list(index_temp_list), max(1, round(len(temp_list) * 0.15)), replace=False)
#             for word_i in index_masc:
#                 temp_list[word_i] = self.tokenizer.tokenizer.mask_token
#             target_sentences_mlm_temp.append(' '.join(temp_list))
#         target_sentences_mlm = target_sentences_mlm_temp.copy()
#         del target_sentences_mlm_temp

        # # Sparc
        # source_sentences_true = source_sentences_mlm.copy()
        # target_sentences_mlm_temp = []
        # for sentence in tqdm(target_sentences_mlm, desc="data mlm"):
        #     temp_list = sentence.split()
        #     kol_mask = round(len(temp_list) * 0.15)
        #     for word_i in range(len(temp_list)):
        #         if ':' in temp_list[word_i]:
        #             temp_list[word_i] = self.tokenizer.tokenizer.mask_token
        #             kol_mask -= 1
        #         if kol_mask == 0:
        #             break
        #     target_sentences_mlm_temp.append(' '.join(temp_list))
        # target_sentences_mlm = target_sentences_mlm_temp.copy()
        # del target_sentences_mlm_temp

        # NSP

        sep_border = int(len(target_sentences_shift) * 0.5)

        sep = randint(1, sep_border)
        source_sentences_shift = source_sentences_shift[sep:sep_border] + source_sentences_shift[
                                                                          :sep] + source_sentences_shift[sep_border:]

        class_label = [[0] for _ in range(sep_border)]
        for _ in range(len(source_sentences_shift) - sep_border):
            class_label.append([1])

        # Add schema

        source_kb_id_mlm = prepare_sql_input(source_kb_id_mlm, self.db2attr_dict)
        source_sentences_mlm = [i + j for i, j in zip(source_sentences_mlm, source_kb_id_mlm)]

        source_kb_id_shift = prepare_sql_input(source_kb_id_shift, self.db2attr_dict)
        source_sentences_shift = [i + j for i, j in zip(source_sentences_shift, source_kb_id_shift)]

        # DataLoader

        tokenized_source_sentences_mlm = [self.tokenizer(i, j) for i, j in
                                          tqdm(zip(source_sentences_mlm, target_sentences_mlm), desc="Tokenizer mlm")]
        tokenized_source_sentences_shift = [self.tokenizer(i, j) for i, j in
                                            tqdm(zip(source_sentences_shift, target_sentences_shift), desc="Tokenizer shift")]
        
        # MLM
        tokenized_target_sentences_mlm = copy.deepcopy(tokenized_source_sentences_mlm)
        for i in tqdm(range(len(tokenized_source_sentences_mlm)), desc="Added mask"):
            ind_q = (tokenized_source_sentences_mlm[i]['token_type_ids'][0] == 1).nonzero(as_tuple=True)[0][:-1]
            index_masc = np.random.choice(list(ind_q), max(1, round(len(ind_q) * 0.15)), replace=False)
            for word_i in index_masc:
                tokenized_source_sentences_mlm[i]['input_ids'][0][word_i] = self.tokenizer.tokenizer.mask_token_id
        
        # Dataset

        dataset_mlm = MTDataset_mlm(tokenized_source_list=tokenized_source_sentences_mlm,
                                    tokenized_target_list=tokenized_target_sentences_mlm, device=self.device)

        dataset_shift = MTDataset_shift(tokenized_source_list=tokenized_source_sentences_shift,
                                        tokenized_target_list=class_label, device=self.device)

        # DataLoader
        
        dataloader_mlm = DataLoader(dataset_mlm, shuffle=True,
                                    batch_size=self.config["batch_size"], drop_last=drop_last)

        dataloader_shift = DataLoader(dataset_shift, shuffle=True,
                                      batch_size=self.config["batch_size"], drop_last=drop_last)

        return dataloader_mlm, dataloader_shift


class DataManager:
    def __init__(self, config, device, tokenizer=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        self.db2attr_dict = json.load(open(os.path.join(self.config["path_repository"],
                                                        self.config["path_data_schema"]), 'r', encoding="utf8"))

        if self.tokenizer is None:
            self.tokenizer = UNIFTokenizer(path_tok=self.config["path_repository"] + self.config["path_query_vocab"],
                                           pre_train_name=self.config["pre_train_tokenizer"],
                                           pad_flag=True,
                                           max_length=self.config["max_sent_len"])

    def prepare_data(self, path_data, drop_last=False):

        dev_data = json.load(open(os.path.join(path_data), 'r', encoding="utf-8"))
        target_sentences = []
        source_sentences = []
        kb_id_sentences = []
        for sample in tqdm(dev_data[:self.config["separate_batch"]], desc="Pars data"):
            target_sentences.append(sample['masked_query'])
            source_sentences.append(sample['question'])
            kb_id_sentences.append(sample['kb_id'])

        # Add schema

        kb_id_sentences = prepare_sql_input(kb_id_sentences, self.db2attr_dict)
        source_sentences = [i + j for i, j in zip(source_sentences, kb_id_sentences)]

        # DataLoader

        tokenized_source_sentences = [self.tokenizer.tkr(i) for i in source_sentences]
        tokenized_target_sentences = [self.tokenizer.tkr(i) for i in target_sentences]

        dataset = MTDataset(tokenized_source_list=tokenized_source_sentences,
                            tokenized_target_list=tokenized_target_sentences, device=self.device)

        dataloader = DataLoader(dataset, shuffle=True, batch_size=self.config["batch_size"], drop_last=drop_last)

        return dataloader


import yaml


def prepare_sql_input(kb_id_list: list[str], db2attr_dict: dict[str:list[str]]) -> list[int]:
    input_list = []
    for kb_id in kb_id_list:
        final_input_str = ' [schema] '
        question_relevant_db_attributes = db2attr_dict[kb_id]
        final_input_str = final_input_str + " ".join(question_relevant_db_attributes)
        input_list.append(final_input_str)
    return input_list

def get_schema_string(table_json):
    """Returns the schema serialized as a string."""
    table_id_to_column_names = collections.defaultdict(list)
    for table_id, name in table_json["column_names_original"]:
        table_id_to_column_names[table_id].append(name.lower())
    tables = table_json["table_names_original"]

    table_strings = []
    for table_id, table_name in enumerate(tables):
        column_names = table_id_to_column_names[table_id]
        table_string = "| %s : %s" % (table_name.lower(), " , ".join(column_names))
        table_strings.append(table_string)
    result_string = "".join(table_strings).lower().replace('\t', "").strip()
    return result_string


if __name__ == "__main__":

    path = "D:/ФизТех/Учеба в маге/2 курс (10сем)/Кафедральные/Машинное обучение продвинутый уровень/статья/Project_cod/"
    data_config = yaml.load(open(path + "configs/data_config.yaml", 'r'), Loader=yaml.Loader)
    data_config["path_repository"] = path
    dm = DataManager_pretrain(data_config, "cpu")

    dataloader_mlm, dataloader_shift, str_true = dm.prepare_data(path_data=path + "data/dev_split.json",
                                                                 drop_last=False)

    for batch, s_t in zip(dataloader_mlm, str_true):
        print(batch[0][0], batch[1][0], batch[2][0], batch[3][0])
        print(dm.tokenizer.decode(batch[0][0]), dm.tokenizer.decode(batch[3][0]))
        print(s_t)
        break

    for batch in dataloader_shift:
        print(batch[0][0], batch[1][0], batch[2][0], batch[3][0])
        print(dm.tokenizer.decode(batch[0][0]))
        break

    print("OK")

    # path = "D:/ФизТех/Учеба в маге/2 курс (10сем)/Кафедральные/Машинное обучение продвинутый уровень/статья/Project_cod/"
    # data_config = yaml.load(open(path + "configs/data_config.yaml", 'r'), Loader=yaml.Loader)
    # data_config["path_repository"] = path
    # dm = DataManager(data_config, "cpu")
    #
    # dataloader = dm.prepare_data(path_data=path + "data/russian_dev_split.json", drop_last=False)
    #
    # for batch in dataloader:
    #     print(batch[0][0], batch[1][0], batch[2][0], batch[3][0])
    #     break
