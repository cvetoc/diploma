from tqdm import tqdm

class Trainer:
    def __init__(self, model, model_config, logger, prin):
        self.model = model
        self.epoch_num = model_config['epoch_num']
        self.logger = logger

        self.prin = prin

        self.logger.log(model_config)

    def train(self, train_dataloader, val_dataloader):
        try:
            iterations = tqdm(range(self.epoch_num))
            iterations.set_postfix({'Score class': 0.0, 'Score str': 0.0, 'Score mask': 0.0,
                                    "train loss mlm": 0.0, "train loss class": 0.0,
                                    "val loss mlm": 0.0, "val loss class": 0.0})
            for epoch in iterations:
                train_epoch_loss_mlm = 0
                train_epoch_loss_clas = 0
                self.model.train()
                for batch_mlm, batch_clas in zip(*train_dataloader):
                    train_loss_mlm, train_loss_clas = self.model.training_step(batch_mlm, batch_clas)
                    train_epoch_loss_mlm += train_loss_mlm
                    train_epoch_loss_clas += train_loss_clas
                train_epoch_loss_mlm = train_epoch_loss_mlm / len(train_dataloader[0])
                train_epoch_loss_clas = train_epoch_loss_clas / len(train_dataloader[1])

                val_epoch_loss_mlm, val_epoch_acc = 0, 0
                val_epoch_loss_clas, val_epoch_clas = 0, 0
                str_score_batch, clas_score_batch, mask_score_batch = 0, 0, 0
                self.model.eval()
                for batch_mlm, batch_clas in zip(*val_dataloader):
                    val_loss = self.model.validation_step_mlm(batch_mlm)
                    val_epoch_loss_mlm += val_loss
                    val_loss = self.model.validation_step_clas(batch_clas)
                    val_epoch_loss_clas += val_loss

                    predicted_samples, _ = self.model.forward(batch_mlm)
                    str_score, actual_sentences, predicted_sentences = self.model.eval_str(predicted_samples,
                                                                                            batch_mlm[-1])
                    str_score_batch += str_score

                    mask_score = self.model.eval_mlm(batch_mlm[0], predicted_samples, batch_mlm[-1])
                    mask_score_batch += mask_score

                    _, predicted_clas = self.model.forward(batch_clas)
                    clas_score = self.model.eval_clas(predicted_clas, batch_clas[-1])
                    clas_score_batch += clas_score

                val_epoch_loss_mlm = val_epoch_loss_mlm / len(val_dataloader[0])
                val_epoch_loss_clas = val_epoch_loss_clas / len(train_dataloader[1])
                str_score_batch = str_score_batch / len(val_dataloader[0])
                mask_score_batch = mask_score_batch / len(val_dataloader[0])
                clas_score_batch = clas_score_batch / len(val_dataloader[1])

                iterations.set_postfix({'Score class': clas_score_batch, 'Score str': str_score_batch, 'Score mask': mask_score_batch,
                                        "train loss mlm": train_epoch_loss_mlm, "train loss class": train_epoch_loss_clas,
                                        "val loss mlm": val_epoch_loss_mlm, "val loss class": val_epoch_loss_clas})

                if self.prin:
                    for a, b in zip(actual_sentences[:5], predicted_sentences[:5]):
                        print(f"{a} ---> {b}")
                    print('##############################')

                self.logger.log({"loss_val_mlm": val_epoch_loss_mlm,
                                 "loss_val_class": val_epoch_loss_clas,
                                 "loss_train_mlm": train_epoch_loss_mlm,
                                 "loss_train_class": train_epoch_loss_clas,
                                 "score_str": str_score_batch,
                                 "score_clas": clas_score_batch,
                                 "score_mask": mask_score_batch})

        except KeyboardInterrupt:
            pass

        print(f"Last {epoch} epoch train loss mlm: ", train_epoch_loss_mlm)
        print(f"Last {epoch} epoch train loss class: ", train_epoch_loss_clas)
        print(f"Last {epoch} epoch val loss mlm: ", val_epoch_loss_mlm)
        print(f"Last {epoch} epoch val loss class: ", val_epoch_loss_clas)
        print(f"Last {epoch} epoch val score str: ", str_score_batch)
        print(f"Last {epoch} epoch val score clas: ", clas_score_batch)
        print(f"Last {epoch} epoch val score mask: ", mask_score_batch)
