import torch.nn as nn
import torch
import math

import src.metrics as metrics
from transformers import BertModel, XLMRobertaForCausalLM, AutoConfig
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from src.models.positional_encoding import PositionalEncoding


class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 device,
                 tokenizer,
                 model_path="bert-base-uncased",
                 lr=0.0001,
                 sched_step=4,
                 sched_gamma=0.1):

        super(Seq2SeqTransformer, self).__init__()
        self.model = BertModel.from_pretrained(model_path).to(device)
        # config = AutoConfig.from_pretrained(model_path)
        # config.is_decoder = True
        # self.model = XLMRobertaForCausalLM.from_pretrained(model_path, config=config).to(device)

        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=8, dim_feedforward=512)
        decoder_norm = nn.LayerNorm(768)
        self.decoder = nn.TransformerDecoder(decoder_layer, 8, decoder_norm)  # , num_layers,

        # self.transformer = nn.Transformer(d_model=768, nhead=8, num_encoder_layers=4,
        #                                   num_decoder_layers=4, dim_feedforward=512, dropout=0.1)

        self.device = device
        self.tokenizer = tokenizer
        self.lr = lr
        self.sched_step = sched_step
        self.sched_gamma = sched_gamma

        self.model.resize_token_embeddings(len(self.tokenizer))

        #         self.linear_mlm = nn.Linear(768, len(self.tokenizer), bias=True)
        #         self.linear_nsp = nn.Linear(768, 2, bias=True)

        #         self.embedding_src = nn.Embedding(len(self.tokenizer), 768)
        self.embedding_trg = nn.Embedding(len(self.tokenizer), 768)
        self.pos_encoding = PositionalEncoding(5000, 768)
        self.linear_src = nn.Linear(768, len(self.tokenizer), bias=False)
        torch.nn.init.xavier_uniform(self.linear_src.weight)

        # сброс пораметров
        # with torch.no_grad():
        #     for name, param in self.named_parameters():
        #         param.copy_(torch.randn(param.size()))

        # for i in [*self.model.named_parameters()]:
        #     i[1].requires_grad = False

        self.trg_mask = self.generate_square_subsequent_mask(0, 0).to(device)
        self.src_mask = self.generate_square_subsequent_mask(0, 1).to(device)

        self.optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=sched_step, gamma=sched_gamma)
        # self.scheduler = self.get_linear_schedule_with_warmup(
        #     self.optimizer, num_warmup_steps=sched_step, num_training_steps=9*sched_step
        # )
        self.loss_tok = nn.CrossEntropyLoss()  # (ignore_index=tokenizer.tokenizer.pad_token_id)
        self.loss_clas = nn.CrossEntropyLoss()

    def reset_learn(self):
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.sched_step,
                                                         gamma=self.sched_gamma)

    def get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        """
        # https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/optimization.py#L108
        #
        Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
        a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

        Args:
            optimizer ([`~torch.optim.Optimizer`]):
                The optimizer for which to schedule the learning rate.
            num_warmup_steps (`int`):
                The number of steps for the warmup phase.
            num_training_steps (`int`):
                The total number of training steps.
            last_epoch (`int`, *optional*, defaults to -1):
                The index of the last epoch when resuming training.

        Return:
            `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
        """

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )

        return LambdaLR(optimizer, lr_lambda, last_epoch)

    def generate_square_subsequent_mask(self, sz, flag):
        if flag == 0:
            return torch.triu(torch.full((sz, sz), float('-inf'), device=self.device), diagonal=1)
        else:
            # TODO: torch.zero
            return torch.triu(torch.full((sz, sz), 0., device=self.device), diagonal=1)

    def forward_generation(self, batch):
        src, attention_mask, trg, trg_am = batch
        trg = trg[:, :-1]
        trg_am = trg_am[:, :-1]

        if self.trg_mask.size(0) != trg.size(1):
            self.trg_mask = self.generate_square_subsequent_mask(trg.size(1), 0)

        emb_trg = self.pos_encoding(self.embedding_trg(trg) * math.sqrt(768)).permute(1, 0, 2)
        hid_src = self.model(input_ids=src, attention_mask=attention_mask).last_hidden_state.permute(1, 0, 2)
        predicted = self.decoder(emb_trg, hid_src, tgt_mask=self.trg_mask, tgt_key_padding_mask=trg_am,
                                 memory_key_padding_mask=attention_mask)
        predicted = self.linear_src(predicted).permute(1, 0, 2)

        return predicted

    #         if self.src_mask.size(0) != src.size(1):
    #             self.src_mask = self.generate_square_subsequent_mask(src.size(1), 1)
    #         if self.trg_mask.size(0) != trg.size(1):
    #             self.trg_mask = self.generate_square_subsequent_mask(trg.size(1), 0)

    #         src = self.pos_encoding(self.embedding_src(src) * math.sqrt(768)).permute(1, 0, 2)
    #         trg = self.pos_encoding(self.embedding_trg(trg) * math.sqrt(768)).permute(1, 0, 2)

    #         output = self.transformer(src, trg, tgt_mask = self.trg_mask) #, tgt_key_padding_mask = trg_am.permute(1, 0), src_key_padding_mask=attention_mask.permute(1, 0))
    #         output = self.linear_src(output).permute(1, 0, 2)

    #         return output

    # output = self.model(input_ids=src, attention_mask=attention_mask, labels=trg)
    # # topi = torch.argmax(output.logits, dim=-1)
    # return output.logits, output

    def forward(self, batch):
        src, token_type_ids, attention_mask, _ = batch

        predicted = self.model(input_ids=src, token_type_ids=token_type_ids, attention_mask=attention_mask)
        decoder_outputs = self.linear_mlm(predicted.last_hidden_state)
        class_outputs = self.linear_nsp(predicted.pooler_output)

        return decoder_outputs, class_outputs

    def training_step(self, batch_mlm, batсh_nsp):
        self.optimizer.zero_grad()

        # mlm loss
        X_tensor, _, _, Y_tensor = batch_mlm
        decoder_outputs, _ = self.forward(batch_mlm)
        decoder_outputs = decoder_outputs.view(-1, decoder_outputs.size(-1))
        decoder_input = X_tensor.view(-1)
        labels = Y_tensor.view(-1)
        # TODO учить на новых токенах
        labels = torch.where(decoder_input == self.tokenizer.tokenizer.mask_token_id, labels, -100)
        # https://discuss.huggingface.co/t/bertformaskedlm-s-loss-and-scores-how-the-loss-is-computed/607/2
        loss_mlm = self.loss_tok(decoder_outputs, labels)

        # nsp loss
        # TODO возможно косяк так как плохо обучается + есть переобучение (видно на val) (возможно в данных лик)
        _, _, _, clas = batсh_nsp
        _, class_outputs = self.forward(batсh_nsp)
        class_outputs = class_outputs.view(-1, class_outputs.size(-1))
        clas = clas.view(-1)
        loss_nsp = self.loss_clas(class_outputs, clas)

        loss = loss_mlm + loss_nsp

        loss.backward()
        self.optimizer.step()
        return loss_mlm.item(), loss_nsp.item()

    def validation_step_mlm(self, batch):
        X_tensor, _, _, Y_tensor = batch
        decoder_outputs, _ = self.forward(batch)
        decoder_outputs = decoder_outputs.view(-1, decoder_outputs.size(-1))
        decoder_input = X_tensor.view(-1)
        labels = Y_tensor.view(-1)
        labels = torch.where(decoder_input == self.tokenizer.tokenizer.mask_token_id, labels, -100)
        loss = self.loss_tok(decoder_outputs, labels)
        return loss.item()

    def validation_step_clas(self, batch):
        X_tensor, _, _, clas = batch
        _, class_outputs = self.forward(batch)
        class_outputs = class_outputs.view(-1, class_outputs.size(-1))
        clas = clas.view(-1)
        loss = self.loss_clas(class_outputs, clas)
        return loss.item()

    def training_step_seq2seq(self, batch):
        self.optimizer.zero_grad()

        X_tensor, _, Y_tensor, _ = batch
        decoder_outputs = self.forward_generation(batch)
        decoder_outputs = decoder_outputs.reshape(-1, decoder_outputs.size(-1))
        labels = Y_tensor[:, 1:].reshape(-1)
        labels = torch.where(labels != self.tokenizer.tokenizer.pad_token_id, labels, -100)
        loss = self.loss_tok(decoder_outputs * 10, labels)  # температура 10

        # _, decoder_outputs = self.forward_generation(batch)
        # loss = decoder_outputs.loss

        loss.backward()
        # temp_norm_grad = torch.nn.utils.clip_grad_norm_(self.parameters(), 4.0)
        # if temp_norm_grad.item() > 40:
        #     print(str(temp_norm_grad) + '______grad______')
        #     print(batch[0])
        #     print('______________________')
        #     print(batch[2])
        # for i in self.parameters():
        #     print(i)
        #     print(i.grad)
        #     break
        self.optimizer.step()
        # print(self.optimizer.param_groups[0]['lr'])
        return loss.item()

    def validation_step_seq2seq(self, batch):
        X_tensor, _, Y_tensor, _ = batch
        decoder_outputs = self.forward_generation(batch)
        decoder_outputs = decoder_outputs.reshape(-1, decoder_outputs.size(-1))
        labels = Y_tensor[:, 1:].reshape(-1)  #
        labels = torch.where(labels != self.tokenizer.tokenizer.pad_token_id, labels, -100)
        loss = self.loss_tok(decoder_outputs * 10, labels)

        # _, decoder_outputs = self.forward_generation(batch)
        # loss = decoder_outputs.loss
        return loss.item()

    def eval_str(self, predicted_ids_list, target_tensor):
        predicted = predicted_ids_list.clone()
        predicted = torch.argmax(predicted, dim=-1)
        predicted = predicted.detach().cpu().numpy()  # .reshape((-1))
        actuals = target_tensor[:, 1:].detach().cpu().numpy()  # .reshape((-1)) # [:, 1:]
        str_score, actual_sentences, predicted_sentences = metrics.str_scorer(
            predicted=predicted, actual=actuals, target_tokenizer=self.tokenizer
        )
        return str_score, actual_sentences, predicted_sentences

    def eval_mlm(self, input_ids_list, predicted_ids_list, target_tensor):
        predicted = predicted_ids_list.clone()
        predicted = torch.argmax(predicted, dim=-1)
        input_tensor = input_ids_list.squeeze(-1).detach().cpu().numpy()
        predicted = predicted.reshape((-1)).detach().cpu().numpy()
        actuals = target_tensor.reshape((-1)).detach().cpu().numpy()
        mlm_score, actual_sentences, predicted_sentences = metrics.mask_scorer(input_tensor=input_tensor,
                                                                               predicted=predicted, actual=actuals,
                                                                               tokenizer=self.tokenizer
                                                                               )
        return mlm_score, actual_sentences, predicted_sentences

    def eval_clas(self, predicted_ids_list, target_tensor):
        predicted = predicted_ids_list.clone()
        predicted = torch.argmax(predicted, dim=-1)
        predicted = predicted.reshape((-1)).detach().cpu()
        actuals = target_tensor.reshape((-1)).detach().cpu()
        clas_score = metrics.clas_scorer(predicted=predicted, actual=actuals)
        return clas_score
