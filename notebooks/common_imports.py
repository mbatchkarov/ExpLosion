import numpy as np
import pandas as pd
import pdb

import django_standalone
from gui.models import Vectors, Experiment, Results, FullResults

from matplotlib import pylab as plt
import seaborn as sns

from IPython import get_ipython
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = 12, 9  # that's default image size for this 
plt.rcParams['savefig.dpi'] = 2 * plt.rcParams['savefig.dpi']