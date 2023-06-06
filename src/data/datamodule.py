from torch.utils.data import DataLoader
import json
import os
from random import randint

from tqdm import tqdm

from src.data.mt_dataset import MTDataset
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

    def prepare_data(self, path_data, drop_last=False, aug=False):

        self.tokenizer = UNIFTokenizer(path="data/query_vocab.json", pad_flag=True,
                                       max_length=self.config["max_sent_len"])

        dev_data = json.load(open(os.path.join(path_data), 'r', encoding="utf-8"))
        target_sentences = []
        source_sentences = []
        for sample in tqdm(dev_data[:130], desc="Pars data"):
            target_sentences.append(sample['masked_sparql'])
            source_sentences.append(sample['question'])

        if aug:
            tools = Augmetator_func_tool()
            augmentor = MM_Augmentation(tools)
            source_sentences = augmentor.run(source_sentences)

        separate_size = int(len(source_sentences) * self.config["separate_size"])

        target_sentences_mlm = target_sentences[:separate_size]
        source_sentences_mlm = source_sentences[:separate_size]

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

        target_sentences_shift = target_sentences[separate_size:]
        source_sentences_shift = source_sentences[separate_size:]

        sep_border = int(len(target_sentences_shift) * 0.5)

        sep = randint(1, sep_border)
        source_sentences_shift = source_sentences_shift[sep:sep_border] + source_sentences_shift[
                                                                          :sep] + source_sentences_shift[sep_border:]

        class_label = [[0] for _ in range(separate_size)]
        for _ in range(sep_border):
            class_label.append([1])
        for _ in range(len(source_sentences) - sep_border - separate_size):
            class_label.append([0])

        target_sentences_X = target_sentences_mlm + target_sentences_shift
        source_sentences_X = source_sentences_mlm + source_sentences_shift

        tokenized_source_sentences = [(self.tokenizer(i, j), z) for i, j, z in
                                      zip(source_sentences_X, target_sentences_X, class_label)]
        tokenized_target_sentences = [self.tokenizer(i, j) for i, j in zip(source_sentences, target_sentences)]

        train_dataset = MTDataset(tokenized_source_list=tokenized_source_sentences,
                                  tokenized_target_list=tokenized_target_sentences, dev=self.device)

        train_dataloader = DataLoader(train_dataset, shuffle=True,
                                      batch_size=self.config["batch_size"], drop_last=drop_last)

        return train_dataloader
