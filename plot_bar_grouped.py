import matplotlib.pyplot as plt
import numpy as np

def main():

    path = 'freq2.txt'
    cols = (1,2)
    path = 'freq5.txt'
    cols = (1,2,3,4,5)

    with open(path) as f:
        # a = np.loadtxt(f, usecols=(1,2)).transpose()
        a = np.loadtxt(f, usecols=cols)
    with open(path) as f:
        # xticks = [line.split()[0] for line in f.readlines()]
        legend = [line.split()[0] for line in f.readlines()]

    legend = [_ + ' ' + {
        'F20': 'Schizophrenia',
        'F25': 'Schizoaffective disorders',
        'F31': 'Bipolar affective disorder',
        'F33': 'Major depressive disorder',
        'I10-I16': 'Hypertensive diseases',
        'I20-I25': 'Ischemic heart diseases',
        'I30-I52': 'Other forms of heart disease',
        'J40-J47': 'Chronic lower respiratory diseases',
        'J60-J70': 'Lung diseases due to external agents',
        'M50-M54': 'Other dorsopathies',
        'F01-F09': 'Mental disorders due to known physiological conditions',
        'J40-J47_and_J60-J70': 'Respiratory diseases',
        'E00-E07': 'Disorders of thyroid gland',
        'K55-K64': 'Other diseases of intestines',
        'M45-M49': 'Spondylopathies',
        'M70-M79': 'Other soft tissue disorders',
        'N17-N19': 'Acute kidney failure and chronic kidney disease',
        'F10-F19': 'Mental and behavioral disorders due to psychoactive substance use',
        }[_] for _ in legend]

    print(a)

    # legend = ['Cluster ' + str(i) for i in range(len(a))]
    xticks = ['Cluster ' + str(i + 1) for i in range(a.shape[1])]

    path = 'plot_bar_grouped.png'

    ylabel = 'Frequency within cluster'

    plot(a, legend, path, ylabel, xticks)

    return

def plot(
    a,
    legend,
    path,
    ylabel,
    xticks,
    # colors = ('#a6cee3', '#1f78b4', '#b2df8a', '#33a02c'),
    # colors = ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd','#ccebc5','#ffed6f'],
    # colors = 2 * ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999'],
    colors = ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd','#ccebc5','#ffed6f', '#ffffff', '#000000', '#cccccc'],
    ):

    # assert len(xticks) == array.shape[0]
    # assert len(legend) == array.shape[1]

    fig, ax = plt.subplots(figsize=(16 / 1.5, 9 / 1.5))

    n = a.shape[0]

    x = np.arange(a.shape[1])

    width = 1 / (n + 1)

    for i in range(n):
        color = colors[i]
        y = a[i]
        ax.bar(x + (i - n/2 + 0.5) * width, y, width, color=color, edgecolor='black')

    plt.xticks(x, xticks, fontsize='small')
    ax.tick_params(labelrotation=45)

    plt.ylabel(ylabel)

    # plt.legend(legend)

    plt.legend(legend, loc='upper left', bbox_to_anchor=(1.04,1), borderaxespad=0)

    fig.savefig(path, bbox_inches="tight")

    print(path)

    return

if __name__ == '__main__':
    main()