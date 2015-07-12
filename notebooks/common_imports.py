import random
import numpy as np
import pandas as pd
import pdb
import django_standalone
from gui.models import Vectors, Experiment, Results, FullResults, Expansions, Clusters

from matplotlib import pylab as plt
import seaborn as sns
from gui.output_utils import get_cv_scores_many_experiment
from gui.user_code import get_demsar_params, pretty_names
from gui.constants import CLASSIFIER, METRIC_DB, BOOTSTRAP_REPS

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
    Compare the scores of pairs of experiment ids and plot a bar chart of the difference in performance.
    Can be a bit hard to read though.
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
                                                                             'expansions__vectors__id',
                                                                             'expansions__vectors__composer',
                                                                             'expansions__vectors__algorithm',
                                                                             'expansions__vectors__dimensionality'])
        diffs.append(mean_scores[0] - mean_scores[1])
        if significance_df is None:
            continue
        if significance_df.significant[0]:
            labels[i] += '*'
    df = pd.DataFrame(dict(Model=labels, Delta=diffs))
    order = df.Model[df.Delta.argsort()].tolist()  # seaborn doesn't like DataFrame-s here
    print(order)
    g = sns.factorplot('Model', 'Delta', data=df, kind='bar',
                       x_order=order if sort_by_magnitude else None,
                       aspect=1.5)
    g.set_xticklabels(rotation=rotation)
    # remove axis labels
    for ax in g.axes.flat:
        ax.set(xlabel='', ylabel='')


def diff_plot_bar(lists, list_ids, xticks,
                  rotation=0, xlabel='', ylabel='Accuracy',
                  hue_order=None, hline_at=None):
    """
    Compare the scores of paired of experiment ids and plot a bar chart or their accuracies.
    :param list1, list2: [1,2,3], [4,5,6] means exp 1 is compared to exp 4, etc ...
    :param list1_id, list2_id: name for the first/ second group of experiments, will appear in legend
    :param xticks: labels for the x-axis, one per pair of experiments, e.g.
    list('abc') will label the first pair 'a', etc. Will appear as ticks on x axis.
    If only two lists are provided a significance test is run for each pair and a * is added if pair is
    significantly different
    :param rotation: angle of x axis ticks
    :param hue_order: order of list1_id, list2_id
    :param hline_at: draw a horizontal line at y=hline_at. Useful for baselines, etc
    """
    assert len(set(map(len, lists))) == 1
    assert len(list_ids) == len(lists)

    df_scores, df_reps, df_groups, df_labels = [], [], [], []
    if len(lists) == 2:
        for i, (a, b) in enumerate(zip(*lists)):
            significance_df, names, mean_scores = get_demsar_params([a, b],
                                                                    name_format=['id',
                                                                                 'expansions__vectors__id',
                                                                                 'expansions__vectors__composer',
                                                                                 'expansions__vectors__algorithm',
                                                                                 'expansions__vectors__dimensionality'])
            if significance_df is None:
                continue
            if significance_df.significant[0]:
                xticks[i] += '*'

    for i, exp_ids in enumerate(zip(*lists)):
        data, folds = get_cv_scores_many_experiment(exp_ids)
        df_scores.extend(data)
        df_reps.extend(folds)
        df_labels.extend(len(folds) * [xticks[i]])
        for list_id in list_ids:
            df_groups.extend(len(folds) // len(lists) * [list_id])

    df = pd.DataFrame(dict(Accuracy=df_scores, reps=df_reps, Method=df_groups, labels=df_labels))
    g = sns.factorplot(y='Accuracy', hue='Method', x='labels', data=df, kind='bar', aspect=1.5, hue_order=hue_order);
    g.set_xticklabels(rotation=rotation);
    # remove axis labels
    for ax in g.axes.flat:
        ax.set(xlabel=xlabel, ylabel=ylabel)
    if hline_at is not None:
        plt.axhline(hline_at, color='black')


def dataframe_from_exp_ids(ids, fields_to_include, abbreviate=True):
    """
    Extracts performance results for given experiments into a long-form
    DataFrame suitable for seaborn.
    :param ids: list of ids to extract
    :param fields_to_include: dict column_name_in_df -> django_query_to_get, e.g.
    {'algo':'expansions__vectors__algorithm', 'comp':'expansions__vectors__composer'}. The DF
    in this example will have 4 columns, [score, folds, comp, algo]
    :param abbreviate: whether to run names of method through the abbreviation map at constants.ABBREVIATIONS
    :return:
    """
    data = {}
    scores, folds = get_cv_scores_many_experiment(ids)
    data['Accuracy'] = scores
    data['folds'] = folds

    for col_name, long_name in fields_to_include.items():
        param_values = pretty_names(ids, [long_name], abbreviate=abbreviate)
        data[col_name] = np.repeat(param_values, len(folds) // len(param_values))

    for col_name, values in data.items():
        print('%s has %d values' % (col_name, len(values)))
    df = pd.DataFrame(data)
    return df[df.Accuracy > -1]  # remove experiments where results are missing


def sort_df_by(df, by):
    """
    Returns the order of items in column `by` in long-form DF that would sort
    the DF by mean accuracy across folds. Useful for seaborn's x_order, hue_order, etc
    :param df:
    :param by:
    :return:
    """
    mean_scores = df.groupby(by).Accuracy.mean()
    return list(mean_scores.index[mean_scores.argsort()])


def random_vect_baseline(corpus='amazon_grouped-tagged'):
    r_id = Experiment.objects.get(expansions__vectors__algorithm='random_vect',
                                  labelled=corpus).id
    return Results.objects.get(id=r_id, classifier=CLASSIFIER).accuracy_mean


def nondistributional_baseline(corpus='amazon_grouped-tagged',
                               document_features_tr='J+N+AN+NN',
                               document_features_ev='AN+NN', **kwargs):
    res = Experiment.objects.get(labelled=corpus,
                                 document_features_tr=document_features_tr,
                                 document_features_ev=document_features_ev,
                                 expansions__decode_handler='BaseFeatureHandler')
    return Results.objects.get(id=res.id, classifier=CLASSIFIER).accuracy_mean


def settings_of(eid, exclude=[]):
    """
    Returns a dict of the settings needed to query for an experiment, e.g.
    >>> s = settings_of(21)
    >>> Experiment.objects.get(**s).id == 21

    Useful in two cases:
    1) get very similar experiment
    >>> s['clusters__num_clusters'] = 200
    >>> Experiment.objects.get(**s)
    2) use GUI to find ID of an interesting experiment, then dump settings into a program
    :param eid: experiment ID
    :param exclude: fields to drop
    :return:
    """
    from copy import deepcopy

    e = Experiment.objects.get(id=eid)
    settings = deepcopy(Experiment.objects.filter(id=eid).values()[0])

    def _add_vectors_settings(nested_keys, keyword):
        settings.update({'%s__vectors__%s' % (keyword, k): v for k, v in nested_keys.items()})
        del settings['%s__vectors__id' % keyword]
        try:
            del settings['%s__vectors_id' % keyword]
        except KeyError:
            pass
        del settings['%s__vectors__path' % keyword]
        del settings['%s__vectors__size' % keyword]
        del settings['%s__vectors__modified' % keyword]

    if e.expansions:
        nested_keys = Expansions.objects.filter(id=e.expansions.id).values()[0]
        settings.update({'expansions__%s' % k: v for k, v in nested_keys.items()})
        del settings['expansions__id']
        if e.expansions.vectors:
            nested_keys = Vectors.objects.filter(id=e.expansions.vectors.id).values()[0]
            _add_vectors_settings(nested_keys, 'expansions')

    if e.clusters:
        nested_keys = Clusters.objects.filter(id=e.clusters.id).values()[0]
        settings.update({'clusters__%s' % k: v for k, v in nested_keys.items()})
        del settings['clusters__id']
        del settings['clusters__vectors_id']
        del settings['clusters__path']
        if e.clusters.vectors:
            nested_keys = Vectors.objects.filter(id=e.clusters.vectors.id).values()[0]
            _add_vectors_settings(nested_keys, 'clusters')

    del settings['expansions_id']
    del settings['clusters_id']
    del settings['id']
    del settings['git_hash']
    del settings['date_ran']
    del settings['minutes_taken']

    for key in exclude:
        try:
            del settings[key]
        except KeyError:
            pass

    return settings


def compare_settings(*ids):
    """
    Comparares the settings of several experiments and prints the differences.
    Useful for when too many experiments are showing up in plots, because the query
    isn't narrow enough

    Example:
    >>> compare_settings(1, 2)
      expansions__vectors__dimensionality         exp 1        exp 2
    0       expansions__vectors__composer  random_neigh  random_vect
    1      expansions__vectors__algorithm  random_neigh  random_vect
    """
    dicts = [settings_of(i) for i in ids]
    data = []
    for key in set().union(*[d.keys() for d in dicts]):
        in_all = all(key in d for d in dicts)
        all_equal = len(set(d.get(key, 'N/A') for d in dicts)) == 1
        if not (in_all and all_equal):
            data.append([key] + [d.get(key, 'N/A') for d in dicts])
        return pd.DataFrame(data, columns=['key'] + ['exp %d' % i for i in ids]).set_index('key')


def sparsify_axis_labels(ax, n=2):
    """
    Sparsify tick labels on the given matplotlib axis, keeping only those whose index is divisible by n
    """
    for idx, label in enumerate(ax.xaxis.get_ticklabels()):
        if idx % n != 0:
            label.set_visible(False)


def compare_neighbours(vectors, names, words=[]):
    """
    Compare the neighbours of several entries in several thesauri
    :param vectors: list of vectors to look up entries in
    :param names: pretty (human-readable) names for the vectors
    :param words: entries to compare. If none are specified, a random sample of
    10 unigrams is selected
    :return:
    """
    if not words:
        words = random.sample([x for x in vectors[0].keys() if not x.count('_')], 10)
    data = []
    for w in words:
        this_row = []
        for v in vectors:
            neigh = v.get_nearest_neighbours(w)
            if neigh:
                this_row.append(', '.join(n[0] for n in neigh[:4]))
            else:
                this_row.append(None)
        data.append(this_row)
    return pd.DataFrame(data, index=words, columns=names)


def my_bootstrap(*args, **kwargs):
    return np.vstack(args)
