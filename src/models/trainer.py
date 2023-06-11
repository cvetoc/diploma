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
            iterations.set_postfix({'Current class': 0.0, 'Current Accuracy': 0.0,
                                    "train loss mlm": 0.0, "train loss class": 0.0,
                                    "val loss mlm": 0.0, "val loss class": 0.0})
            for epoch in iterations:
                train_epoch_loss_mlm = 0
                train_epoch_loss_clas = 0
                self.model.train()
                for batch_mlm, batch_clas in zip(*train_dataloader):
                    train_loss = self.model.training_step_mlm(batch_mlm)
                    train_epoch_loss_mlm += train_loss
                    train_loss = self.model.training_step_clas(batch_clas)
                    train_epoch_loss_clas += train_loss
                train_epoch_loss_mlm = train_epoch_loss_mlm / len(train_dataloader[0])
                train_epoch_loss_clas = train_epoch_loss_clas / len(train_dataloader[1])

                val_epoch_loss_mlm, val_epoch_acc = 0, 0
                val_epoch_loss_clas, val_epoch_clas = 0, 0
                mask_score_batch, clas_score_batch = 0, 0
                self.model.eval()
                for batch_mlm, batch_clas in zip(*val_dataloader):
                    val_loss = self.model.validation_step_mlm(batch_mlm)
                    val_epoch_loss_mlm += val_loss
                    val_loss = self.model.validation_step_clas(batch_clas)
                    val_epoch_loss_clas += val_loss

                    predicted_samples, _ = self.model.forward(batch_mlm)
                    mask_score, actual_sentences, predicted_sentences = self.model.eval_mlm(predicted_samples,
                                                                                            batch_mlm[-1])
                    mask_score_batch += mask_score

                    _, predicted_clas = self.model.forward(batch_clas)
                    clas_score = self.model.eval_clas(predicted_clas, batch_clas[-1])
                    clas_score_batch += clas_score

                print("mask_score: "+str(mask_score)+" clas_score: "+str(clas_score))
                val_epoch_loss_mlm = val_epoch_loss_mlm / len(val_dataloader[0])
                val_epoch_loss_clas = val_epoch_loss_clas / len(train_dataloader[1])
                mask_score_batch = mask_score_batch / len(val_dataloader[0])
                clas_score_batch = clas_score_batch / len(val_dataloader[1])

                iterations.set_postfix({'Current class': clas_score_batch, 'Current Accuracy': mask_score_batch,
                                        "train loss mlm": train_epoch_loss_mlm, "train loss class": train_epoch_loss_clas,
                                        "val loss mlm": val_epoch_loss_mlm, "val loss class": val_epoch_loss_clas})

                if self.prin:
                    for a, b in zip(actual_sentences[:5], predicted_sentences[:5]):
                        print(f"{a} ---> {b}")
                    print('##############################')

                self.logger.log({"val_loss_mlm": val_epoch_loss_mlm,
                                 "val_loss_class": val_epoch_loss_clas,
                                 "train_loss_mlm": train_epoch_loss_mlm,
                                 "train_loss_class": train_epoch_loss_clas,
                                 "acc_score": mask_score_batch,
                                 "clas_score": clas_score_batch})

        except KeyboardInterrupt:
            pass

        print(f"Last {epoch} epoch train loss mlm: ", train_epoch_loss_mlm)
        print(f"Last {epoch} epoch train loss class: ", val_epoch_loss_clas)
        print(f"Last {epoch} epoch val loss mlm: ", val_epoch_loss_mlm)
        print(f"Last {epoch} epoch val loss class: ", train_epoch_loss_clas)
        print(f"Last {epoch} epoch val mask: ", mask_score)
        print(f"Last {epoch} epoch val clas: ", clas_score)
