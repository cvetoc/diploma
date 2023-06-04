from transformers import BertTokenizer


class RUTokenizer:
    def __init__(self, pad_flag, max_length):

        self.tokenizer = BertTokenizer.from_pretrained('cointegrated/rubert-tiny')
        self.pad_token_id = self.tokenizer.pad_token_id

        self.special_tokens_set = self.tokenizer.special_tokens_map
        self.max_sent_len = max_length
        self.pad_flag = pad_flag

    def pad_sent(self, token_ids_list):
        if len(token_ids_list) < self.max_sent_len:
            padded_token_ids_list = token_ids_list + [self.pad_token_id] * (self.max_sent_len - len(token_ids_list))
        else:
            padded_token_ids_list = token_ids_list[:self.max_sent_len - 1] + [self.tokenizer.sep_token_id]
        return padded_token_ids_list

    def __call__(self, text_list):
        token_list =self.tokenizer(text_list, return_tensors="pt", truncation=True,
                                   max_length=self.max_sent_len)['input_ids'][0]
        if self.pad_flag:
            token_list = self.pad_sent(token_list.tolist())

        return token_list

    def __len__(self):
        return len(self.tokenizer.vocab)


    def decode(self, token_list):
        return self.tokenizer.decode(token_list)