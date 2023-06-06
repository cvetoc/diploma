import torch
from torch.utils.data import Dataset


class MTDataset(Dataset):
    def __init__(self, tokenized_source_list, tokenized_target_list, dev):
        self.tokenized_source_list = tokenized_source_list
        self.tokenized_target_list = tokenized_target_list
        self.device = dev

    def __len__(self):
        return len(self.tokenized_source_list)

    def __getitem__(self, idx):
        # TODO UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True),
        #  rather than torch.tensor(sourceTensor).
        #   torch.tensor(self.tokenized_source_list[idx][0]["token_type_ids"][0]).to(self.device),

        source_ids, target_ids, token_type_ids, attention_mask, class_label = (
            torch.tensor(self.tokenized_source_list[idx][0]["input_ids"][0]).to(self.device),
            torch.tensor(self.tokenized_target_list[idx]["input_ids"][0]).to(self.device),
            torch.tensor(self.tokenized_source_list[idx][0]["token_type_ids"][0]).to(self.device),
            torch.tensor(self.tokenized_source_list[idx][0]["attention_mask"][0]).to(self.device),
            torch.tensor(self.tokenized_source_list[idx][1]).to(self.device)
        )
        return source_ids, target_ids, token_type_ids, attention_mask, class_label
