import torch
from torch.utils.data import Dataset


class MTDataset_mlm(Dataset):
    def __init__(self, tokenized_source_list, tokenized_target_list, dev):
        self.tokenized_source_list = tokenized_source_list
        self.tokenized_target_list = tokenized_target_list
        self.device = dev

    def __len__(self):
        return len(self.tokenized_source_list)

    def __getitem__(self, idx):
        source_ids, token_type_ids, attention_mask, target_ids = (
            torch.tensor(self.tokenized_source_list[idx]["input_ids"][0]).to(self.device),
            torch.tensor(self.tokenized_source_list[idx]["token_type_ids"][0]).to(self.device),
            torch.tensor(self.tokenized_source_list[idx]["attention_mask"][0]).to(self.device),
            torch.tensor(self.tokenized_target_list[idx]["input_ids"][0]).to(self.device)
        )
        return source_ids, token_type_ids, attention_mask, target_ids


class MTDataset_shift(Dataset):
    def __init__(self, tokenized_source_list, tokenized_target_list, dev):
        self.tokenized_source_list = tokenized_source_list
        self.tokenized_target_list = tokenized_target_list
        self.device = dev

    def __len__(self):
        return len(self.tokenized_source_list)

    def __getitem__(self, idx):
        source_ids, token_type_ids, attention_mask, class_label = (
            torch.tensor(self.tokenized_source_list[idx]["input_ids"][0]).to(self.device),
            torch.tensor(self.tokenized_source_list[idx]["token_type_ids"][0]).to(self.device),
            torch.tensor(self.tokenized_source_list[idx]["attention_mask"][0]).to(self.device),
            torch.tensor(self.tokenized_target_list[idx]).to(self.device)
        )
        return source_ids, token_type_ids, attention_mask, class_label


class MTDataset(Dataset):
    def __init__(self, tokenized_source_list, tokenized_target_list, dev):
        self.tokenized_source_list = tokenized_source_list
        self.tokenized_target_list = tokenized_target_list
        self.device = dev

    def __len__(self):
        return len(self.tokenized_source_list)

    def __getitem__(self, idx):
        # source_ids, token_type_ids, attention_mask, target_ids, target_attention_mask = (
        #     torch.tensor(self.tokenized_source_list[idx]["input_ids"][0]).to(self.device), # TODO: drop to device
        #     torch.tensor(self.tokenized_source_list[idx]["token_type_ids"][0]).to(self.device),
        #     torch.tensor(self.tokenized_source_list[idx]["attention_mask"][0]).to(self.device),
        #     torch.tensor(self.tokenized_target_list[idx]["input_ids"][0]).to(self.device),
        #     torch.tensor(self.tokenized_target_list[idx]["attention_mask"][0]).to(self.device),
        # )
        # return source_ids, token_type_ids, attention_mask, target_ids, target_attention_mask

        source_ids, source_attention_mask, target_ids, target_attention_mask = (
            torch.tensor(self.tokenized_source_list[idx]["input_ids"][0]).to(self.device),
            torch.tensor(self.tokenized_source_list[idx]["attention_mask"][0], dtype=torch.float).to(self.device),
            torch.tensor(self.tokenized_target_list[idx]["input_ids"][0]).to(self.device),
            torch.tensor(self.tokenized_target_list[idx]["attention_mask"][0], dtype=torch.float).to(self.device)
        )
        return source_ids, source_attention_mask, target_ids, target_attention_mask


class MTDataset_HF(Dataset):
    def __init__(self, tokenized_source_list, tokenized_target_list, device):
        self.tokenized_source_list = tokenized_source_list
        self.tokenized_target_list = tokenized_target_list
        self.device = device

    def __len__(self):
        return len(self.tokenized_source_list)

    def __getitem__(self, idx):
        # source_ids, token_type_ids, attention_mask, target_ids, target_attention_mask = (
        #     torch.tensor(self.tokenized_source_list[idx]["input_ids"][0]).to(self.device), # TODO: drop to device
        #     torch.tensor(self.tokenized_source_list[idx]["token_type_ids"][0]).to(self.device),
        #     torch.tensor(self.tokenized_source_list[idx]["attention_mask"][0]).to(self.device),
        #     torch.tensor(self.tokenized_target_list[idx]["input_ids"][0]).to(self.device),
        #     torch.tensor(self.tokenized_target_list[idx]["attention_mask"][0]).to(self.device),
        # )
        # return source_ids, token_type_ids, attention_mask, target_ids, target_attention_mask

        source_ids, source_attention_mask, target_ids = (
            torch.tensor(self.tokenized_source_list[idx]["input_ids"][0]).to(self.device),
            torch.tensor(self.tokenized_source_list[idx]["attention_mask"][0], dtype=torch.float).to(self.device),
            torch.tensor(self.tokenized_target_list[idx]["input_ids"][0]).to(self.device)
        )
        return {'input_ids': source_ids, 'attention_mask': source_attention_mask, 'labels': target_ids}