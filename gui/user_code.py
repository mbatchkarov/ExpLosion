from io import BytesIO
import base64
from itertools import combinations
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import pandas as pd
from critical_difference.plot import do_plot, print_figure
from gui.constants import METRIC_DB, CLASSIFIER, ABBREVIATIONS, BOOTSTRAP_REPS, SIGNIFICANCE_LEVEL
from gui.models import Experiment, Results, get_ci, memory
from thesisgenerator.utils.conf_file_utils import parse_config_file

config, _ = parse_config_file('conf/output.conf')


def get_tables(exp_ids):
    res = []
    if not exp_ids:
        return res
    if config['performance_table']:
        res.append(get_performance_table(exp_ids, ci=config['performance_ci'])[0])
    if config['significance_table']:
        res.append(get_demsar_params(exp_ids)[0])
    return res


def get_static_figures(exp_ids):
    if config['static_figures'] and exp_ids:
        return ["static/figures/stats-exp%d-0.png" % n for n in exp_ids]
    else:
        return []


def get_generated_figures(exp_ids):
    # todo there must be a better way to do this, as it conflicts with ipython notebook
    import matplotlib as mpl

    mpl.use('Agg')  # for running on headless servers
    if not exp_ids:
        return []
    res = []
    if config['significance_diagram']:
        res.append(figure_to_base64(get_demsar_diagram(*get_demsar_params(exp_ids))))
    return res


def populate_manually():
    # run manually in django console to populate the database
    if Experiment.objects.count():
        return

    with open('experiments.txt') as infile:
        table_descr = [x.strip() for x in infile.readlines()]
    # '5,gigaw,R2,0,Add,AN_NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler' ,
    # '136,word2vec,MR,100,Right,AN_NN,word2vec,0,1,0,SignifiedOnlyFeatureHandler',
    for line in table_descr:
        num, unlab, lab, svd, comp, doc_feats, thes_feats, baronified, \
        use_similarity, use_random_neighbours, decode_handler = line.split(',')
        exp = Experiment(id=num, composer=comp, labelled=lab,
                         unlabelled=unlab, svd=svd,
                         document_features=doc_feats,
                         thesaurus_features=thes_feats,
                         baronified=bool(int(baronified)),
                         use_similarity=use_similarity,
                         use_random_neighbours=use_random_neighbours,
                         decode_handler=decode_handler)
        exp.save()


def figure_to_base64(fig):
    canvas = FigureCanvas(fig)
    s = BytesIO()
    canvas.print_png(s)
    return base64.b64encode(s.getvalue())


def pretty_names(exp_ids, name_format=['expansions__vectors__algorithm', 'expansions__vectors__composer'],
                 abbreviate=True):
    names = []
    for eid in exp_ids:
        e = Experiment.objects.values_list(*name_format).get(id=eid)
        this_name = '-'.join((str(ABBREVIATIONS.get(x, x)) for x in e)) if abbreviate else '-'.join(e)
        names.append(this_name)
    return names


def get_demsar_params(exp_ids, name_format=['expansions__vectors__algorithm', 'expansions__vectors__composer']):
    """
    Gets parameters for `get_demsar_diagram`. Methods whose results are not in DB are dropped silently
    :param exp_ids: ids of experiments to look up
    :param name_format: how to form a human-readable summary of each of the competing methods.
    The fields listed here are extracted from their DB entry and concatenated. For example, ['id'] will name
    each method with its 'id' field, and ['vectors__algorithm'] will name each method with the value of
    `vectors.algorithm`
    :return: tuple of:
                - pandas.DataFrame of significance test
                - names of the methods compared, as specified by `name_format`
                - mean scores of the methods compared
    """
    mean_scores = [foo.accuracy_mean for foo in Results.objects.filter(id__in=exp_ids, classifier=CLASSIFIER)]

    names = dict(zip(exp_ids, pretty_names(exp_ids, name_format)))

    data = []
    for a, b in combinations(exp_ids, 2):
        obs_diff, pval = pairwise_randomised_significance(get_data_for_signif_test(a),
                                                          get_data_for_signif_test(b))
        data.append((names[a], Results.objects.get(id=a, classifier=CLASSIFIER).accuracy_mean,
                     names[b], Results.objects.get(id=b, classifier=CLASSIFIER).accuracy_mean,
                    obs_diff, pval, pval < SIGNIFICANCE_LEVEL))
    sign_table = pd.DataFrame(data, columns='name1 acc1 name2 acc2 mean_diff pval significant'.split())
    return sign_table, np.array(names), np.array(mean_scores)


def get_demsar_diagram(significance_df, names, mean_scores, filename=None):
    """

    :param significance_df: pd.DataFrame made out of statsmodels significance table
    :param names: names of competing methods, len(names) == len(mean_scores)
    :param mean_scores: mean scores of each competing methods (one per method)
    :param filename: where to output image to
    :return:
    """
    if significance_df is None:
        return None
    idx = np.argsort(mean_scores)
    mean_scores = list(mean_scores[idx])
    names = list(names[idx])

    def get_insignificant_pairs(*args):
        mylist = []
        for a, b, significant_diff in zip(significance_df.group1,
                                          significance_df.group2,
                                          significance_df.significant):
            if significant_diff == 'False':
                mylist.append((names.index(a), names.index(b)))
        return sorted(set(tuple(sorted(x)) for x in mylist))

    fig = do_plot(mean_scores, get_insignificant_pairs, names, arrow_vgap=0.15, link_vgap=.08,
                  link_voffset=0.1, xlabel=METRIC_DB)
    fig.set_canvas(plt.gcf().canvas)
    if not filename:
        filename = "sign-%s.pdf" % ('_'.join(sorted(set(names))))[:200]  # there's a limit on that
    print('Saving figure to %s' % filename)
    print_figure(fig, filename, format='pdf', dpi=300)
    return fig


def get_performance_table(exp_ids, ci=False):
    print('running performance query for experiments %s' % exp_ids)
    all_data = []
    if len(exp_ids) != len(set(exp_ids)):
        raise ValueError('DUPLICATE EXPERIMENTS: got %s, unique: %s' % (exp_ids - set(exp_ids)))
    for exp_id in exp_ids:
        composer_name = '%s-%s' % (exp_id, get_composer_name(exp_id))
        result = Results.objects.get(id=exp_id, classifier=CLASSIFIER)
        if not result:
            # table or result does not exist
            print('Missing results entry for exp %d and classifier %s' % (exp_id, CLASSIFIER))
            continue

        if ci:
            score_mean, score_low, score_high, _ = get_ci(exp_id, clf=CLASSIFIER)
        else:
            score_mean = Results.objects.get(id=exp_id, classifier=CLASSIFIER).accuracy_mean
            score_low, score_high = -1, -1

        vectors = Experiment.objects.get(id=exp_id).expansions.vectors
        row = [exp_id, str(vectors), CLASSIFIER, composer_name, score_mean, score_low, score_high]
        all_data.append(row if ci else row[:-2])

    header = ['exp id', 'vectors', 'classifier', 'composer', METRIC_DB] + (['low', 'high'] if ci else [])
    table = pd.DataFrame(all_data, columns=header)
    return table, exp_ids


def get_data_for_signif_test(exp_id, clf=CLASSIFIER):
    y = Results.objects.get(id=exp_id, classifier=clf).predictions
    y_gold = Results.objects.get(id=exp_id, classifier=clf).gold

    # check gold standard matches right answers
    # assert set(y) == set(y_gold) # cant assert that, a classifier may never predict a class
    assert len(y) == len(y_gold)
    return np.concatenate([y, y_gold])


def pairwise_significance_exp_ids(ids, name_format=['id']):
    """
    Computes significance between predefined pairs of experiments
    :param ids: list of pairs of experiments ids to be compared [(e1, e2), (e1, e3), (e3, e4)]
    :param name_format:
    """
    tables = []
    for pair in ids:
        print('Running significance for', pair)
        tables.append(get_demsar_params(pair, name_format=name_format)[0])
    return pd.concat(tables, ignore_index=True)



@memory.cache
def pairwise_randomised_significance(y, z, nboot=BOOTSTRAP_REPS, statistic=accuracy_score):
    """
    Start off with the predictions and gold stands answers for two models, A and B, on some dataset.

           Gold  Pred
    model
    A         0     0
    A         1     0
    -----------------
    B         0     0
    B         0     0
    B         1     0
    B         1     1

    Here A has an accuracy of 0.5 and B has an accuracy of 0.75. Is this significantly different?
    Note: The predictions of A and B may differ in length in my case because the vocabulary of the document classifier
    is dependent on the vector set used, and as a result some documents may be left with no features at decode time.
    Which document this is may differ between models A and B. For taking into account the document system A cannot
    classify, the above table may be written as:

               Gold  Pred
    model
    A         0     0
    A         -     -
    A         -     -
    A         1     0
    -----------------
    B         0     0
    B         0     0
    B         1     0
    B         1     1

    However, in that case the test statistic (e.g. accuracy) is ill-defined, se so happily drop the two middle
    data points (documents).

    To test for if the difference is significant, we permute the predictions of A and B. If the two arrays were of the
    same lenght, we could have gone over them and swapped element-wise with a probability of 0.5. In the case where they
    are not we can shuffle the predictions array, keeping the right number of predictions for A and B. We need to
    also shuffle the gold-standard array the exact same way, e.g.

               Gold  Pred
    model
    A         0     0
    A         1     1<-------
    -----------------       |
    B         0     0       |
    B         0     0       |
    B         1     0       |
    B         1     0<-------

    Here we did a very light shuffle, swapping the second and last rows. We now compute the test statistic (e.g.
    accuracy) for both models again, and record the difference. If B is genuinely better, we should expect B's score to
    drop after the shuffle, because it got and is penalised for some of A's poor predictions. The opposite is true for
    A. As we permute, the observed difference in accuracy should drop e.g. 95% of the time if B is genuinely better.
    That is our p-value.

    See also:
    https://stat.duke.edu/~ar182/rr/examples-gallery/PermutationTest.html
    http://stackoverflow.com/a/24801874/419338

    :param y: concatenated gold standard labels and predictions for model A, e.g. [0100]. This model made 2 predictions
    and has an accuracy of 0.5
    :type y: np.array
    :param z: gold standard and predictions for model B. These may be different to the ones for model A, e.g.
    [00110001]. This model made 4 prediction and has an accuracy of 0.75.
    :type z: np.array
    :param nboot: number of boostrap iterations
    :param statistic: function to invoke on gold standard and predictions, e.g. sklearn's `accuracy_score`. This is also
    the test statistic
    :return: (difference_in_mean_values_of_statistic, pvalue)
    """
    half_y = len(y) / 2
    y_gold = y[half_y:]
    y = y[:half_y]

    half_z = len(z) / 2
    z_gold = z[half_z:]
    z = z[:half_z]

    y_acc = statistic(y_gold, y)
    z_acc = statistic(z_gold, z)
    theta_hat = np.abs(y_acc - z_acc)  # observed difference in test statistic between A and B
    estimates = []  # estimated difference in test statistic between A and B for each bootstrap iteration
    pooled_predictions = np.hstack([y, z]).ravel()
    pooled_gold = np.hstack([y_gold, z_gold]).ravel()
    for _ in range(nboot):
        perm_ind = np.arange(len(pooled_predictions))
        np.random.shuffle(perm_ind)
        pooled_predictions = pooled_predictions[perm_ind]
        pooled_gold = pooled_gold[perm_ind]

        y_acc = statistic(pooled_gold[:half_y], pooled_predictions[:half_y])
        z_acc = statistic(pooled_gold[half_y:], pooled_predictions[half_y:])
        new_diff = np.abs(y_acc - z_acc)
        # print('New difference', new_diff)
        estimates.append(new_diff)
    diff_count = len(np.where(estimates <= theta_hat)[0])
    hat_asl_perm = 1.0 - (diff_count / nboot)
    return theta_hat, hat_asl_perm
    # statsmodels multiple comparison insists this is shaped like this: (statistic, p-value)


def get_composer_name(n):
    vectors = Experiment.objects.get(id=n).expansions.vectors
    return vectors.composer if vectors else None
