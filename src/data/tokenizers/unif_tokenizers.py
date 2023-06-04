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
        token_list = self.tokenizer.decode(token_list)
        return token_list
