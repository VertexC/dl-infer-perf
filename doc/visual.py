import os
import pandas as pd
# libraries
import numpy as np
import matplotlib.pyplot as plt


def grouped_barplot(data, title, filename):
    fig = plt.figure()
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
    plt.show()

    plt.savefig(filename)


def visualize(directory, batch_size):
    dfs = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(directory, filename), index_col=None)
            dfs.append(df)
    frame = pd.concat(dfs, axis=0, ignore_index=True)

    batchf = frame.loc[frame['batch_size'] == batch_size]

    for model in set(batchf['model']):
        modelf = batchf.loc[batchf['model'] == model]
        # data: (optimizer, fe): time
        data = {}
        for _, row in modelf.iterrows():
            optimizer = row['optimizer']
            if pd.isnull(optimizer):
                optimizer = ''
            data[(optimizer, row['fe'])] = row['time']
        print(model, batch_size, '\n', data)
        title = '{}-{}'.format(model, batch_size)
        grouped_barplot(data, title,
                        os.path.join(directory, '{}.png'.format(title)))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="benchmark visualizer")
    parser.add_argument("dir", help="dir of benchmark result")
    parser.add_argument("--batch", type=int, default=64)

    args = parser.parse_args()
    visualize(args.dir, args.batch)
