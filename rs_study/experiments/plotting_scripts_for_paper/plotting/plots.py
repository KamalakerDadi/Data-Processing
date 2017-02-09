"""Run stripplot using seaborn for pair-wise comparisons
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def stripplot_to_pdf(data, save_path, x=None, y=None, hue=None,
                     style='whitegrid', fontsize=2, rows=1, cols=1,
                     figsize=(4, 4), **kwargs):
    """ Data plotted as stripplot using seaborn and saved in a pdf
    given in save_path

    Parameters
    ----------
    data : pd.DataFrame or path to csv file
        single or list of data to plot into pdf.

    save_path : str
        Path to save the pdf plot.

    """
    if isinstance(data, basestring):
        data = pd.read_csv(data)

    if isinstance(data, (list, tuple)):
        cols = len(data)

    if not isinstance(data, (list, tuple)):
        data = [data, ]

    sns.set_style(style)
    sns.set(font_scale=fontsize)

    with PdfPages(save_path) as pdf:
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize,
                                 squeeze=True, sharey=True)
        axes = axes.reshape(-1)
        for ax, d in zip(axes, data):
            sns.stripplot(x=x, y=y, hue=hue, data=d, ax=ax, **kwargs)
        pdf.savefig(fig)
        plt.close()
