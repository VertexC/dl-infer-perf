import os
import pandas as pd
# libraries
import numpy as np
import matplotlib.pyplot as plt

MODEL_NUM = 4


def do_subplot(model_num, dir_num):
    axs = []

    def _do_subplot(i, model_i):
        if i == 0:
            ax = plt.subplot(model_num, dir_num, model_i * dir_num + i + 1)
            axs.append(ax)
        else:
            plt.subplot(model_num,
                        dir_num,
                        model_i * dir_num + i + 1,
                        sharey=axs[model_i])

    return _do_subplot


def grouped_barplot(plotter, i, model_i, data, title):
    plotter(i, model_i)
    labels = set()
    groups = set()
    for (label, group), _ in data.items():
        labels.add(label)
        groups.add(group)
    groups = list(sorted([g for g in groups]))
    labels = list(sorted([l for l in labels]))
    values = []
    for label in labels:
        vals = []
        for group in groups:
            if (label, group) in data:
                vals.append(data[(label, group)])
            else:
                vals.append(0)
        values.append(vals)

    # set width of bar
    barWidth = 0.15

    # Set position of bar on X axis
    rs = [[i * barWidth * (len(labels) + 1) for i in range(len(groups))]]
    for i in range(len(labels) - 1):
        r = [x + barWidth for x in rs[-1]]
        rs.append(r)

    color = list('rgbkymc')
    colors = [color[i % len(color)] for i in range(len(labels))]

    print(rs, '\n', values, labels, colors)
    # Make the plot
    for r, vals, label, color in zip(rs, values, labels, color):
        if label == '':
            label = 'baseline'
        bars = plt.bar(r,
                       vals,
                       color=color,
                       width=barWidth,
                       edgecolor='white',
                       label=label)
        for bar in bars:
            yval = bar.get_height()
            pos = yval + .005
            if yval == 0:
                yval = 'nan'
            else:
                yval = '{:.2f}'.format(yval)
            plt.text(bar.get_x() + 0.03, pos, yval)
    # Add xticks on the middle of the group bars
    plt.xlabel('fe', fontweight='bold')
    plt.xticks([r + 1.5 * barWidth for r in rs[0]], groups)

    # Create legend & Show graphic
    plt.title(title)
    plt.legend()


def visualize(source, fes, models, optimizers, plotter, group_i, group, batch_size):
    print(fes, models, optimizers)
    frame = pd.read_csv(source, index_col=None)
    batchf = frame.loc[frame['group'] == group]
    batchf = batchf.loc[batchf['batch_size'] == batch_size]
    for model_i, model in enumerate(models):
        # data: {(fe, optimizer): metric}
        # FIXME: what is there is no model
        modelf = batchf.loc[batchf['model'] == model]

        data = {}
        for _, row in modelf.iterrows():
            optimizer = row['optimizer']
            fe = row['fe']
            if fe in fes and optimizer in optimizer:
                data[(optimizer, row['fe'])] = row['metric']
        # place holders
        for optimizer in optimizers:
            for fe in fes:
                comb = (optimizer, fe)
                if comb not in data:
                    data[comb] = 0
        title = '{}-{}'.format(model, batch_size)
        print(data)
        grouped_barplot(plotter, group_i, model_i, data, title)


def scan_parameters(source, groups):
    df = pd.read_csv(source, index_col=None)
    frame = df.loc[np.isin(df['group'], groups)]
    models = sorted(list(set(frame['model'])))
    optimizers = sorted(list(set(frame['optimizer'])))
    fes = sorted(list(set(frame['fe'])))
    return fes, models, optimizers


def visualize_group(source, groups, out, model_filter, opt_filter, fe_filter, batch_size):
    fes, models, optimizers = scan_parameters(source, groups)
    fig = plt.figure(figsize=(25, 25))
    plt.suptitle(' Vs '.join([g for g in groups]), fontsize=40)
    fig.tight_layout()
    if model_filter is not None and len(model_filter) > 0:
        models = model_filter
    if opt_filter is not None and len(opt_filter) > 0:
        optimizers = opt_filter
    if fe_filter is not None and len(fe_filter) > 0:
        fes = fe_filter
    plotter = do_subplot(len(models), len(groups))
    for i, g in enumerate(groups):
        visualize(source, fes, models, optimizers, plotter, i, g, batch_size)
    plt.show()
    plt.savefig('{}-{}'.format(out, batch_size))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="benchmark visualizer")
    parser.add_argument('source', type=str, help='input csv')
    parser.add_argument('group',
                        metavar='N',
                        type=str,
                        nargs='+',
                        help='groups name to show')
    parser.add_argument('out', type=str, help='output result image')
    parser.add_argument("--batch", type=int, default='64')
    parser.add_argument('--model', type=str, nargs='*', help='selected models to visualize')
    parser.add_argument('--opt', type=str, nargs='*', help='selected optimizers to visualize')
    parser.add_argument('--fe', type=str, nargs='*', help='selected front-end to visualize')
    args = parser.parse_args()
    visualize_group(args.source, args.group, args.out, args.model, args.opt, args.fe, args.batch)
