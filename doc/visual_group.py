import os
import pandas as pd
# libraries
import numpy as np
import matplotlib.pyplot as plt

MODEL_NUM = 4


def do_subplot(model_num, dir_num):
    axs = []

    def _do_subplot(dir_i, model_i):
        if dir_i == 0:
            ax = plt.subplot(model_num, dir_num, model_i * dir_num + dir_i + 1)
            axs.append(ax)
        else:
            plt.subplot(model_num,
                        dir_num,
                        model_i * dir_num + dir_i + 1,
                        sharey=axs[model_i])

    return _do_subplot


def grouped_barplot(plotter, dir_i, model_i, data, title):
    plotter(dir_i, model_i)
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


def visualize(fes, models, optimizers, plotter, dir_i, directory, batch_size):
    dfs = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(directory, filename), index_col=None)
            dfs.append(df)
    frame = pd.concat(dfs, axis=0, ignore_index=True)

    batchf = frame.loc[frame['batch_size'] == batch_size]
    for model_i, model in enumerate(models):
        # data: {(fe, optimizer): metric}
        # FIXME: what is there is no model
        modelf = batchf.loc[batchf['model'] == model]
        data = {}
        for _, row in modelf.iterrows():
            optimizer = row['optimizer']
            data[(optimizer, row['fe'])] = row['metric']
        # place holders
        for optimizer in optimizers:
            for fe in fes:
                comb = (optimizer, fe)
                if comb not in data:
                    data[comb] = 0
        title = '{}-{}'.format(model, batch_size)
        grouped_barplot(plotter, dir_i, model_i, data, title)


def scan_parameters(dirs):
    dfs = []
    for d in dirs:
        for filename in os.listdir(d):
            if filename.endswith('.csv'):
                df = pd.read_csv(os.path.join(d, filename), index_col=None)
                dfs.append(df)
    frame = pd.concat(dfs, axis=0, ignore_index=True)
    models = sorted(list(set(frame['model'])))
    optimizers = sorted(list(set(frame['optimizer'])))
    fes = sorted(list(set(frame['fe'])))
    return fes, models, optimizers


def visualize_group(dirs, out, batch_size):
    fes, models, optimizers = scan_parameters(dirs)
    fig = plt.figure(figsize=(25, 25))
    fig.tight_layout()
    plotter = do_subplot(len(models), len(dirs))
    for i, d in enumerate(dirs):
        visualize(fes, models, optimizers, plotter, i, d, batch_size)
    plt.show()
    plt.savefig(out)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="benchmark visualizer")
    parser.add_argument('dirs',
                        metavar='N',
                        type=str,
                        nargs='+',
                        help='dirs of benchmark result')
    parser.add_argument('out', type=str, help='output result image')
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    visualize_group(args.dirs, args.out, args.batch)
