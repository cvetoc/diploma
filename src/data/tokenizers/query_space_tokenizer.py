import json


class QuerySpaceTokenizer:
    def __init__(self, query_list, vocab, pad_flag, shift=0):
        self.pad_flag = pad_flag
        self.word2index = {}
        self.word2count = {}
        self.index2word = {2: "[CLS]", 3: "[SEP]", 1: "[UNK]", 0: "[PAD]", 4: "[MASK]"}
        self.word2index = {"[CLS]": 2, "[SEP]": 3, "[UNK]": 1, '[PAD]': 0, "[MASK]": 4}
        self.n_words = shift
        self.max_sent_len = -1
        self.special_tokens_set = {'[CLS]', '[SEP]', '[PAD]'}

        assert type(vocab) in [list, str]
        if type(vocab) is str:
            token_list = json.load(open(vocab, 'r'))
            self.load_vocab(token_list)
        elif type(vocab) is list:
            self.load_vocab(vocab)

        for sent in query_list:
            sent_words_amount = len(sent.split())
            if sent_words_amount > self.max_sent_len:
                self.max_sent_len = sent_words_amount

        self.max_sent_len += 2  # always add CLS/EOS

        print(f'Code tokenizer fitted - {len(self.word2index)} tokens')

    def load_vocab(self, token_list):
        for token in [" "]+token_list:
            if token not in self.word2index:
                self.word2index[token] = self.n_words
                self.word2count[token] = 1
                self.index2word[self.n_words] = token
                self.n_words += 1
            else:
                self.word2count[token] += 1

    def pad_sent(self, token_ids_list):
        if len(token_ids_list) < self.max_sent_len:
            padded_token_ids_list = token_ids_list + [self.word2index['[PAD]']] * (
                        self.max_sent_len - len(token_ids_list))
        else:
            padded_token_ids_list = token_ids_list[:self.max_sent_len - 1] + [self.word2index['[SEP]']]
        return padded_token_ids_list

    def __call__(self, sentence):
        tokenized_data = self.tokenize(sentence)
        if self.pad_flag:
            tokenized_data = self.pad_sent(tokenized_data)
        return tokenized_data

    def tokenize(self, sentence):
        tokenized_data = []
        tokenized_data.append(self.word2index['[CLS]'])
        for word in sentence.split():
            if word in self.word2index:
                tokenized_data.append(self.word2index[word])
            else:
                tokenized_data.append(self.word2index['[UNK]'])
        tokenized_data.append(self.word2index['[SEP]'])
        return tokenized_data

    def decode(self, token_list):
        predicted_tokens = []

        for token_id in token_list:
            predicted_token = self.index2word[token_id]
            predicted_tokens.append(predicted_token)
        filtered_tokens = list(filter(lambda x: x not in self.special_tokens_set, predicted_tokens))

        return filtered_tokens

    def __len__(self):
        return len(self.word2index)
