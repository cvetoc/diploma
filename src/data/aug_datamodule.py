from torch.utils.data import DataLoader
import json
import os

from tqdm import tqdm

from src.data.mt_dataset import MTDataset
from src.data.tokenizers.query_space_tokenizer import QuerySpaceTokenizer
from src.data.tokenizers.ru_tokenizer import RUTokenizer
from src.data.mm_augmentation import MM_Augmentation
from src.data.augmetator import Augmetator_func_tool


class DataManager:
    def __init__(self, config, device):
        super().__init__()
        self.source_tokenizer = None
        self.target_tokenizer = None
        self.config = config
        self.device = device

    def prepare_data(self, path_trina, path_val):
        dev_data = json.load(open(os.path.join(path_trina), 'r', encoding="utf-8"))
        target_train_sentences = [sample['masked_sparql'] for sample in dev_data]
        source_train_sentences = [sample['question'] for sample in dev_data]

        dev_data = json.load(open(os.path.join(path_val), 'r', encoding="utf-8"))
        target_val_sentences = [sample['masked_sparql'] for sample in dev_data]
        source_val_sentences = [sample['question'] for sample in dev_data]

        tools = Augmetator_func_tool()
        augmentor = MM_Augmentation(tools)
        source_train_sentences = augmentor.run(source_train_sentences)

        self.source_tokenizer = RUTokenizer(pad_flag=True, max_length=self.config["max_sent_len"])
        tokenized_source_train_sentences = [self.source_tokenizer(s) for s in tqdm(source_train_sentences)]
        tokenized_source_val_sentences = [self.source_tokenizer(s) for s in tqdm(source_val_sentences)]

        self.target_tokenizer = QuerySpaceTokenizer(target_train_sentences, "data/query_vocab.json", pad_flag=True)
        tokenized_target_train_sentences = [self.target_tokenizer(s) for s in tqdm(target_train_sentences)]
        tokenized_target_val_sentences = [self.target_tokenizer(s) for s in tqdm(target_val_sentences)]

        train_dataset = MTDataset(tokenized_source_list=tokenized_source_train_sentences,
                                  tokenized_target_list=tokenized_target_train_sentences, dev=self.device)

        val_dataset = MTDataset(tokenized_source_list=tokenized_source_val_sentences,
                                tokenized_target_list=tokenized_target_val_sentences, dev=self.device)

        train_dataloader = DataLoader(train_dataset, shuffle=True,
                                      batch_size=self.config["batch_size"], )

        val_dataloader = DataLoader(val_dataset, shuffle=True,
                                    batch_size=self.config["batch_size"], drop_last=True)

        return train_dataloader, val_dataloader
