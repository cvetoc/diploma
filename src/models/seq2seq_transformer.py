import torch.nn as nn
import torch
import math

import src.metrics as metrics
from transformers import BertForPreTraining


class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 device,
                 tokenizer,
                 sched_step=20,
                 sched_gamma=0.1,
                 lr=0.001):
        super(Seq2SeqTransformer, self).__init__()
        self.model = BertForPreTraining.from_pretrained("cointegrated/rubert-tiny").to(device)

        self.device = device
        self.tokenizer = tokenizer
        self.lr = lr
        self.sched_step = sched_step
        self.sched_gamma = sched_gamma

        self.model.resize_token_embeddings(len(self.tokenizer))

        # обучить с 0
        with torch.no_grad():
            for name, param in self.named_parameters():
                param.copy_(torch.randn(param.size()))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=sched_step, gamma=sched_gamma)
        self.loss = nn.CrossEntropyLoss()

        self.sm = nn.Softmax(dim=1)

    def reset_learn(self):
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.sched_step, gamma=self.sched_gamma)

    def forward(self, batch):
        src, _, token_type_ids, attention_mask, _ = batch

        predicted = self.model(input_ids=src, token_type_ids=token_type_ids, attention_mask=attention_mask)
        decoder_outputs = torch.argmax(predicted.prediction_logits, dim=-1)
        class_outputs = predicted.seq_relationship_logits

        return decoder_outputs, class_outputs

    def training_step(self, batch):
        self.optimizer.zero_grad()
        X_tensor, Y_tensor, _, _, clas = batch
        decoder_outputs, class_outputs = self.forward(batch)
        labels = Y_tensor
        labels = torch.where(decoder_outputs == self.tokenizer.tokenizer.mask_token_id, Y_tensor, -100)
        loss = self.loss(class_outputs, clas) + self.loss(decoder_outputs.to(torch.float), labels.to(torch.float))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validation_step(self, batch):
        X_tensor, Y_tensor, _, _, clas = batch
        decoder_outputs, class_outputs = self.forward(batch)
        labels = Y_tensor
        labels = torch.where(decoder_outputs == self.tokenizer.tokenizer.mask_token_id, Y_tensor, -100)
        loss = self.loss(class_outputs, clas) + self.loss(decoder_outputs.to(torch.float), labels.to(torch.float))
        return loss.item()

    def eval_bleu(self, predicted_ids_list, target_tensor):
        predicted = predicted_ids_list.clone()
        predicted = predicted.squeeze(-1).detach().cpu().numpy()
        actuals = target_tensor.squeeze(-1).detach().cpu().numpy()
        bleu_score, actual_sentences, predicted_sentences = metrics.bleu_scorer(
            predicted=predicted, actual=actuals, target_tokenizer=self.tokenizer
        )
        return bleu_score, actual_sentences, predicted_sentences

    def eval_clas(self, predicted_ids_list, target_tensor):
        predicted = predicted_ids_list.clone()
        predicted = self.sm(predicted)
        predicted = predicted.squeeze(-1).detach().cpu()
        actuals = target_tensor.squeeze(-1).detach().cpu()
        clas_score = metrics.clas_scorer(predicted=predicted, actual=actuals)
        return clas_score
