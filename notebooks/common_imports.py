import numpy as np
import pandas as pd
import pdb

import django_standalone
from gui.models import Vectors, Experiment, Results, FullResults

from matplotlib import pylab as plt
import seaborn as sns

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