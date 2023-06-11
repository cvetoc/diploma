import ast
from matplotlib import pyplot as plt
from prettytable import PrettyTable


def graf(path):

    data_graf = {'val_loss_mlm': [], 'val_loss_class': [], 'train_loss_mlm': [], 'train_loss_class': [],
                 'acc_score': [], "clas_score": []}

    lebes = ['val_loss_mlm', 'val_loss_class', 'train_loss_mlm', 'train_loss_class', 'acc_score', 'clas_score']

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

    x = range(1, len(data_graf["acc_score"]) + 1)

    ax1.title.set_text('Loss mlm')
    ax1.plot(x, data_graf['val_loss_mlm'], label='val loss mlm')
    ax1.plot(x, data_graf['train_loss_mlm'], label='train loss mlm')
    ax1.legend()

    ax2.title.set_text('Loss class')
    ax2.plot(x, data_graf['val_loss_class'], label='val loss class')
    ax2.plot(x, data_graf['train_loss_class'], label='train loss class')
    ax2.legend()

    ax3.title.set_text('Score class')
    ax4.title.set_text('Score acc')
    ax3.plot(x, data_graf['clas_score'], label='clas score')
    ax4.plot(x, data_graf['acc_score'], label='acc score')
    ax3.legend()
    ax4.legend()

    plt.show()
