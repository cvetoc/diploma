import ast
from matplotlib import pyplot as plt
from prettytable import PrettyTable

# self.logger.log({"loss_val_mlm": val_epoch_loss_mlm,
#                                  "loss_val_class": val_epoch_loss_clas,
#                                  "loss_train_mlm": train_epoch_loss_mlm,
#                                  "loss_train_class": train_epoch_loss_clas,
#                                  "val_score_clas": val_clas_score_batch,
#                                  "val_score_mask": val_mask_score_batch,
#                                  "train_score_clas": train_clas_score_batch,
#                                  "train_score_mask": train_mask_score_batch})

def graf(path):

    # TODO доделать переписать

    data_graf = {'loss_val_mlm': [], 'loss_val_class': [], 'loss_train_mlm': [], 'loss_train_class': [],
                 'val_score_mask': [], 'val_score_clas': [], 'train_score_mask': [], 'train_score_clas': []}

    lebes = list(data_graf.keys())

    with open(path, 'r') as f:
        description_model = f.readline()
        for lin in f:
            line = ast.literal_eval(lin)
            for leb in lebes:
                data_graf[leb].append(line[leb])

    description_model = ast.literal_eval(description_model)

    table_print = PrettyTable()
    table_print.field_names = ["параметры", "значения"]

    for i in description_model:
        table_print.add_row((i, description_model[i]))

    print(table_print)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    fig.set_size_inches(16, 8, forward=True)

    x = range(1, len(data_graf[lebes[0]]) + 1)

    ax1.title.set_text('Loss mlm')
    ax1.plot(x, data_graf['loss_val_mlm'], label='val')
    ax1.plot(x, data_graf['loss_train_mlm'], label='train')
    ax1.legend()

    ax2.title.set_text('Loss class')
    ax2.plot(x, data_graf['loss_val_class'], label='val')
    ax2.plot(x, data_graf['loss_train_class'], label='train')
    ax2.legend()


    ax3.title.set_text('Score mlm')
    ax3.plot(x, data_graf['val_score_mask'], label='val')
    ax3.plot(x, data_graf['train_score_mask'], label='train')
    ax3.legend()

    ax4.title.set_text('Score class')
    ax4.plot(x, data_graf['val_score_clas'], label='val')
    ax4.plot(x, data_graf['train_score_clas'], label='train')
    ax4.legend()

    plt.show()
