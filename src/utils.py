import ast
from matplotlib import pyplot as plt
from prettytable import PrettyTable


def graf(path):

    data_graf = {'loss_val_mlm': [], 'loss_val_class': [], 'loss_train_mlm': [], 'loss_train_class': [],
                 'score_str': [], 'score_clas': [], 'score_mask': []}

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
    ax4.title.set_text('Score class')

    ax3.plot(x, data_graf['score_str'], label='score str')
    ax3.plot(x, data_graf['score_mask'], label='score mask')

    ax4.plot(x, data_graf['score_clas'], label='score class')
    ax3.legend()
    ax4.legend()

    plt.show()
