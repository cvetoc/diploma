from tqdm import tqdm

class Trainer:
    def __init__(self, model, model_config, logger, prin):
        self.model = model
        self.epoch_num = model_config['epoch_num']
        self.logger = logger

        self.prin = prin

        self.logger.log(model_config)

    def pre_train(self, train_dataloader, val_dataloader):
        try:
            iterations = tqdm(range(self.epoch_num))
            iterations.set_postfix({'train score class': 0.0, 'train score mask': 0.0,
                                    'val score class': 0.0, 'val score mask': 0.0,
                                    "train loss mlm": 0.0, "train loss class": 0.0,
                                    "val loss mlm": 0.0, "val loss class": 0.0})
            for epoch in iterations:
                train_epoch_loss_mlm, train_mask_score_batch = 0, 0
                train_epoch_loss_clas, train_clas_score_batch = 0, 0
                self.model.train()
                for batch_mlm, batch_clas in zip(*train_dataloader):
                    train_loss_mlm, train_loss_clas = self.model.training_step(batch_mlm, batch_clas)
                    train_epoch_loss_mlm += train_loss_mlm
                    train_epoch_loss_clas += train_loss_clas

                    predicted_samples, _ = self.model.forward(batch_mlm)

                    mask_score, _, _ = self.model.eval_mlm(batch_mlm[0], predicted_samples, batch_mlm[-1])
                    train_mask_score_batch += mask_score

                    _, predicted_clas = self.model.forward(batch_clas)

                    clas_score = self.model.eval_clas(predicted_clas, batch_clas[-1])


                    train_clas_score_batch += clas_score

                train_epoch_loss_mlm = train_epoch_loss_mlm / len(train_dataloader[0])
                train_epoch_loss_clas = train_epoch_loss_clas / len(train_dataloader[1])
                train_mask_score_batch = train_mask_score_batch / len(train_dataloader[0])
                train_clas_score_batch = train_clas_score_batch / len(train_dataloader[1])

                val_epoch_loss_mlm, val_mask_score_batch = 0, 0
                val_epoch_loss_clas, val_clas_score_batch = 0, 0
                self.model.eval()
                for batch_mlm, batch_clas in zip(*val_dataloader):
                    val_loss = self.model.validation_step_mlm(batch_mlm)
                    val_epoch_loss_mlm += val_loss
                    val_loss = self.model.validation_step_clas(batch_clas)
                    val_epoch_loss_clas += val_loss

                    predicted_samples, _ = self.model.forward(batch_mlm)

                    mask_score, actual_sentences, predicted_sentences = self.model.eval_mlm(batch_mlm[0], predicted_samples, batch_mlm[-1])
                    val_mask_score_batch += mask_score

                    _, predicted_clas = self.model.forward(batch_clas)
                    clas_score = self.model.eval_clas(predicted_clas, batch_clas[-1])
                    val_clas_score_batch += clas_score

                val_epoch_loss_mlm = val_epoch_loss_mlm / len(val_dataloader[0])
                val_epoch_loss_clas = val_epoch_loss_clas / len(val_dataloader[1])
                val_mask_score_batch = val_mask_score_batch / len(val_dataloader[0])
                val_clas_score_batch = val_clas_score_batch / len(val_dataloader[1])

                iterations.set_postfix({'train score class': train_clas_score_batch, 'train score mask': train_mask_score_batch,
                                        'val score class': val_clas_score_batch, 'val score mask': val_mask_score_batch,
                                        "train loss mlm": train_epoch_loss_mlm, "train loss class": train_epoch_loss_clas,
                                        "val loss mlm": val_epoch_loss_mlm, "val loss class": val_epoch_loss_clas})

                if self.prin:
                    for a, b in zip(actual_sentences[:5], predicted_sentences[:5]):
                        print(f"{a} --->\n {b}")
                    print('##############################')

                self.logger.log({"loss_val_mlm": val_epoch_loss_mlm,
                                 "loss_val_class": val_epoch_loss_clas,
                                 "loss_train_mlm": train_epoch_loss_mlm,
                                 "loss_train_class": train_epoch_loss_clas,
                                 "val_score_clas": val_clas_score_batch,
                                 "val_score_mask": val_mask_score_batch,
                                 "train_score_clas": train_clas_score_batch,
                                 "train_score_mask": train_mask_score_batch})

        except KeyboardInterrupt:
            pass

        print(f"Last {epoch} epoch train loss mlm: ", train_epoch_loss_mlm)
        print(f"Last {epoch} epoch train loss class: ", train_epoch_loss_clas)
        print(f"Last {epoch} epoch val loss mlm: ", val_epoch_loss_mlm)
        print(f"Last {epoch} epoch val loss class: ", val_epoch_loss_clas)
        print(f"Last {epoch} epoch train score clas: ", train_clas_score_batch)
        print(f"Last {epoch} epoch train score mask: ", train_mask_score_batch)
        print(f"Last {epoch} epoch val score clas: ", val_clas_score_batch)
        print(f"Last {epoch} epoch val score mask: ", val_mask_score_batch)

    def train(self, train_dataloader, val_dataloader):
        # TODO: допистьа для двойной модели
        try:
            iterations = tqdm(range(self.epoch_num))
            iterations.set_postfix({'train score': 0.0, 'val score': 0.0,
                                    "train loss": 0.0, "val loss": 0.0})
            for epoch in iterations:
                train_epoch_loss, train_score_batch = 0.0, 0.0
                self.model.train()

                for batch in train_dataloader:
                    train_loss = self.model.training_step_seq2seq(batch)
                    train_epoch_loss += train_loss

                    predicted_samples, _ = self.model.forward_generation(batch)

                    mask_score, _, _ = self.model.eval_str(predicted_samples, batch[2])
                    train_score_batch += mask_score

                train_epoch_loss = train_epoch_loss / len(train_dataloader)
                train_score_batch = train_score_batch / len(train_dataloader)

                val_epoch_loss, val_score_batch = 0, 0
                self.model.eval()
                for batch in val_dataloader:
                    val_loss = self.model.validation_step_seq2seq(batch)
                    val_epoch_loss += val_loss

                    predicted_samples, _ = self.model.forward_generation(batch)

                    mask_score, actual_sentences, predicted_sentences = self.model.eval_str(predicted_samples, batch[2])
                    val_score_batch += mask_score

                val_epoch_loss = val_epoch_loss / len(val_dataloader)
                val_score_batch = val_score_batch / len(val_dataloader)

                iterations.set_postfix(
                    {'train score': train_score_batch,
                     'val score': val_score_batch,
                     "train loss": train_epoch_loss,
                     "val loss": val_epoch_loss})

                if self.prin:
                    for a, b in zip(actual_sentences[:5], predicted_sentences[:5]):
                        print(f"{a} ---> {b}")
                    print('##############################')

                # TODO: доделать
                self.logger.log({"loss_val": val_epoch_loss,
                                 "loss_train": train_epoch_loss,
                                 "val_score": val_score_batch,
                                 "train_score": train_score_batch})

        except KeyboardInterrupt:
            pass

        print(f"Last {epoch} epoch train loss: ", train_epoch_loss)
        print(f"Last {epoch} epoch val loss: ", val_epoch_loss)
        print(f"Last {epoch} epoch train score: ", train_score_batch)
        print(f"Last {epoch} epoch val score: ", val_score_batch)
