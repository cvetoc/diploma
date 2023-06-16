import torch.nn as nn
import torch
import math

import src.metrics as metrics
from transformers import BertForPreTraining
from transformers.optimization import Adafactor

class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 device,
                 tokenizer,
                 lr=0.001,
                 sched_step=50,
                 sched_gamma=0.1):

        super(Seq2SeqTransformer, self).__init__()
        self.model = BertForPreTraining.from_pretrained("cointegrated/rubert-tiny").to(device)

        self.device = device
        self.tokenizer = tokenizer
        self.lr = lr
        self.sched_step = sched_step
        self.sched_gamma = sched_gamma

        self.model.resize_token_embeddings(len(self.tokenizer))

        # обучить с 0
        # with torch.no_grad():
        #     for name, param in self.named_parameters():
        #         param.copy_(torch.randn(param.size()))

        self.optimizer = Adafactor(self.model.parameters(), relative_step=True)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=sched_step, gamma=sched_gamma)
        self.loss_tok = nn.CrossEntropyLoss()#(ignore_index=tokenizer.tokenizer.pad_token_id)
        self.loss_clas = nn.CrossEntropyLoss()

        self.sm = nn.Softmax(dim=1)

    def reset_learn(self):
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.sched_step, gamma=self.sched_gamma)

    def forward(self, batch):
        src, token_type_ids, attention_mask, _ = batch

        predicted = self.model(input_ids=src, token_type_ids=token_type_ids, attention_mask=attention_mask)
        decoder_outputs = predicted.prediction_logits
        class_outputs = predicted.seq_relationship_logits

        return decoder_outputs, class_outputs

    def training_step_mlm(self, batch):
        self.optimizer.zero_grad()
        _, _, _, Y_tensor = batch
        decoder_outputs, _ = self.forward(batch)
        decoder_outputs = decoder_outputs.view(-1, decoder_outputs.size(-1))
        labels = Y_tensor.view(-1)
        loss = self.loss_tok(decoder_outputs, labels)
        loss.backward()
        self.optimizer.step()
        #self.scheduler.step()
        return loss.item()

    def training_step_clas(self, batch):
        # TODO возможно косяк так как плохо обучается + есть переобучение (водно на val)
        self.optimizer.zero_grad()
        _, _, _, clas = batch
        _, class_outputs = self.forward(batch)
        class_outputs = class_outputs.view(-1, class_outputs.size(-1))
        clas = clas.view(-1)
        # labels = torch.where(decoder_outputs == self.tokenizer.tokenizer.mask_token_id, Y_tensor, -100)
        loss = self.loss_clas(class_outputs, clas)
        loss.backward()
        self.optimizer.step()
        #self.scheduler.step()
        return loss.item()

    def validation_step_mlm(self, batch):
        X_tensor, _, _, Y_tensor = batch
        decoder_outputs, _ = self.forward(batch)
        decoder_outputs = decoder_outputs.view(-1, decoder_outputs.size(-1))
        labels = Y_tensor.view(-1)
        loss = self.loss_tok(decoder_outputs, labels)
        return loss.item()

    def validation_step_clas(self, batch):
        X_tensor, _, _, clas = batch
        _, class_outputs = self.forward(batch)
        class_outputs = class_outputs.view(-1, class_outputs.size(-1))
        clas = clas.view(-1)
        loss = self.loss_clas(class_outputs, clas)
        return loss.item()

    def eval_str(self, predicted_ids_list, target_tensor):
        predicted = predicted_ids_list.clone()
        predicted = torch.argmax(predicted, dim=-1)
        predicted = predicted.squeeze(-1).detach().cpu().numpy()
        actuals = target_tensor.squeeze(-1).detach().cpu().numpy()
        str_score, actual_sentences, predicted_sentences = metrics.str_scorer(
            predicted=predicted, actual=actuals, target_tokenizer=self.tokenizer
        )
        return str_score, actual_sentences, predicted_sentences

    def eval_mlm(self, input_ids_list, predicted_ids_list, target_tensor):
        predicted = predicted_ids_list.clone()
        predicted = torch.argmax(predicted, dim=-1)
        input_tensor = input_ids_list.squeeze(-1).detach().cpu().numpy()
        predicted = predicted.squeeze(-1).detach().cpu().numpy()
        actuals = target_tensor.squeeze(-1).detach().cpu().numpy()
        bleu_score = metrics.mask_scorer(input_tensor=input_tensor,
            predicted=predicted, actual=actuals, tokenizer=self.tokenizer
        )
        return bleu_score

    def eval_clas(self, predicted_ids_list, target_tensor):
        predicted = predicted_ids_list.clone()
        predicted = torch.argmax(predicted, dim=-1)
        predicted = predicted.squeeze(-1).detach().cpu()
        actuals = target_tensor.squeeze(-1).detach().cpu()
        clas_score = metrics.clas_scorer(predicted=predicted, actual=actuals)
        return clas_score
