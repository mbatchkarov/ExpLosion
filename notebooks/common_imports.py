import numpy as np
import pandas as pd
import pdb
import django_standalone
from gui.models import Vectors, Experiment, Results, FullResults

from matplotlib import pylab as plt
import seaborn as sns
from gui.user_code import get_demsar_params

sns.set_style("whitegrid")
rc = {'xtick.labelsize': 16,
      'ytick.labelsize': 16,
      'axes.labelsize': 18,
      'axes.labelweight': '900',
      'legend.fontsize': 20,
      'font.family': 'cursive',
      'font.monospace': 'Nimbus Mono L',
      'lines.linewidth': 2,
      'lines.markersize': 9,
      'xtick.major.pad': 20}
sns.set_context(rc=rc)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 22

from IPython import get_ipython
try:
    get_ipython().magic('matplotlib inline')
    plt.rcParams['figure.figsize'] = 12, 9  # that's default image size for this
    plt.rcParams['savefig.dpi'] = 2 * plt.rcParams['savefig.dpi']
except AttributeError:
    # when not running in IPython
    pass


def diff_plot(list1, list2, labels, sort_by_magnitude=True, rotation=90):
    """
    Compare the scores of pairs of experiment ids and plot a bar chart
    :param list1, list2: [1,2,3], [4,5,6] means exp 1 is compared to exp 4, etc ...
    :param labels: labels for the x-axis, one per pair of experiments
    :param sort_by_magnitude: if true, pairs on x-axis are sorted by magnitude of
    difference, otherwise by order defined in `ids`
    """
    assert len(list1) == len(list2) == len(labels)
    diffs = []
    for i, (a, b) in enumerate(zip(list1, list2)):
        significance_df, names, mean_scores = get_demsar_params([a, b],
                                                                name_format=['id',
                                                                             'vectors__id',
                                                                             'vectors__composer',
                                                                             'vectors__algorithm',
                                                                             'vectors__dimensionality'])
        diffs.append(mean_scores[0] - mean_scores[1])
        if significance_df is None:
            continue
        if significance_df.significant[0] == 'True':
            labels[i] += '*'
    df = pd.DataFrame(dict(Model=labels, Delta=diffs))
    order = df.Model[df.Delta.argsort()].tolist() # seaborn doesn't like DataFrame-s here
    print(order)
    g = sns.factorplot('Model', 'Delta', data=df, kind='bar',
                       x_order=order if sort_by_magnitude else None,
                       aspect=1.5);
    g.set_xticklabels(rotation=rotation);
    # remove axis labels
    for ax in g.axes.flat:
        ax.set(xlabel='', ylabel='')