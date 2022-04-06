import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def draw_heatmap(threads, chunk_size, data, title, fmt, schedule, size, type, cbarlabel):
    reshape_tuple = len(threads), len(chunk_size)
    fig, ax = plt.subplots()
    plt.title(title, loc='right')
    plt.xlabel("Chunk size")
    plt.ylabel("Number of threads")
    new_data = np.reshape(np.array(data), reshape_tuple)
    im, cbar = heatmap(new_data, threads, chunk_size, ax=ax, cmap='YlGn', cbarlabel=cbarlabel)

    texts = annotate_heatmap(im, valfmt=fmt)

    fig.tight_layout()
    plt.savefig(f'./results/{schedule}/{type}_{size}.png')
    plt.close(fig)


def plot(threads, chunk_size, time, speedup, schedule, size):
    ## TIME
    draw_heatmap(threads, chunk_size, time * (10 ** 3),
                 f"Time for {schedule} with {size} values", "{x:.3f} ms"
                 , schedule, size, "time", "time [ms]")
    ## SPEEDUP
    draw_heatmap(threads, chunk_size, speedup,
                 f"Speedup for {schedule} with {size} values", "{x:.3f}"
                 , schedule, size, "speedup", "speedup value")


def add_speedup(data: pd.DataFrame):
    speedups = []

    times_for_one_proc = data.groupby(['Schedule', 'Size', 'Chunk size']).first()['Time']
    for index, record in data.iterrows():
        time_for_one = times_for_one_proc[record['Schedule']][record['Size']][record['Chunk size']]
        speedups.append(time_for_one / record['Time'])

    data['Speedup'] = speedups


def create_dirs():
    import os

    paths = [
        './results',
        './results/static',
        './results/dynamic',
        './results/guided',
        './results/runtime',
    ]

    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


if __name__ == "__main__":
    data = pd.read_csv('./out.csv')

    create_dirs()

    schedules = data['Schedule'].unique()
    sizes = data['Size'].unique()

    add_speedup(data)

    grouped = data.groupby(['Schedule', 'Size'])

    for schedule in schedules:
        for size in sizes:
            grouping = grouped.get_group((schedule, size))
            plot(grouping['Threads'].unique(), grouping['Chunk size'].unique(), grouping['Time'], grouping['Speedup'],
                 schedule, size)
