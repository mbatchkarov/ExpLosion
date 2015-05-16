from io import BytesIO
import base64
import re
from configobj import ConfigObj
from sklearn.metrics import accuracy_score
import validate
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import MultiComparison
import statsmodels.api as sm
from critical_difference.plot import do_plot, print_figure
from gui.models import Experiment, Table, Results, FullResults, get_ci
from gui.utils import ABBREVIATIONS
from gui.output_utils import get_scores, get_cv_fold_count

CLASSIFIER = 'MultinomialNB'
# the name differs between the DB a the csv files, can't be bothered to fix
# METRIC_DB = 'accuracy'
# METRIC_CSV_FILE = 'accuracy_score'
METRIC_DB = 'macrof1'
METRIC_CSV_FILE = 'macroavg_f1'


def parse_config_file(conf_file):
    # copied from thesisgen
    config = ConfigObj(conf_file, configspec='confrc')
    validator = validate.Validator()
    result = config.validate(validator)
    # if not result:
    # print('Invalid configuration')
    # sys.exit(1)
    return config


config = parse_config_file('output.conf')


def get_tables(exp_ids):
    res = []
    if not exp_ids:
        return res
    if config['performance_table']:
        res.append(get_performance_table(exp_ids)[0])
    if config['significance_table']:
        res.append(get_significance_table(exp_ids)[0])
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


def make_df(table:Table, index_cols=None):
    if table is None or not table.rows:
        return None
    df = pd.DataFrame(table.rows, columns=table.header)
    if index_cols:
        df.set_index(index_cols, inplace=True)
    return df


def get_demsar_params(exp_ids, name_format=['vectors__algorithm', 'vectors__composer']):
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
    table, exp_ids = get_performance_table(exp_ids)
    if not table.rows:
        return None, None, None
    scores_table = make_df(table, 'exp id').convert_objects(convert_numeric=True)

    data, _, exp_ids = get_scores(exp_ids)
    names, full_names = [], []
    for eid in exp_ids:
        e = Experiment.objects.values_list(*name_format).get(id=eid)
        this_name = '-'.join((str(ABBREVIATIONS.get(x, x)) for x in e))
        names.append(this_name)
        cv_folds = get_cv_fold_count([eid])[0]
        full_names.extend([this_name] * cv_folds)

    sign_table, _ = get_significance_table(exp_ids, data=data, names=full_names)
    sign_table = make_df(sign_table)

    return sign_table, np.array(names), np.array(scores_table[METRIC_DB])


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


def get_performance_table(exp_ids):
    print('running performance query for experiments %s' % exp_ids)

    # for exp_list in exp_lists:
    # if isinstance(exp_list, int):
    # exp_list = [exp_list]
    # composer = '%s-%s' % ('-'.join(str(foo) for foo in exp_list),
    # get_composer_name(exp_list[0]))
    # composers.extend([composer] * (cv_folds * len(exp_list)))
    # for exp_number in exp_list:

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

        score_mean, score_low, score_high, _ = get_ci(exp_id, clf=CLASSIFIER)
        vectors = Experiment.objects.get(id=exp_id).vectors
        all_data.append([exp_id, str(vectors), CLASSIFIER, composer_name,
                         score_mean, score_low, score_high])
    table = Table(['exp id', 'vectors', 'classifier', 'composer', METRIC_DB, 'low', 'high'],
                  all_data,
                  'Performance over bootstrapped evaluation')
    return table, exp_ids


def get_significance_table(exp_ids, classifier='MultinomialNB', data=None, names=None):
    """

    :param exp_ids:
    :param classifier:
    :param data: array of all scores (per method, per CV fold)
    :param names: array of methos names that correspond to `data`, eg [a, a, b, b, c, c] for 2 methods, 2 fold CV
    :return: :raise ValueError:
    """
    print('Running significance for experiments %r' % exp_ids)
    if data is None and names is None:
        data, names, exp_ids = get_scores(exp_ids, classifier=classifier)

    if len(set(names)) < 2:
        print('Cannot run significance test on less than 2 methods: %r' % set(names))
        return None, None
    print(data, names, exp_ids)
    mod = MultiComparison(np.array(data), names, group_order=sorted(set(names)))
    # a = mod.tukeyhsd(alpha=0.05)
    a = mod.allpairtest(pairwise_randomised_significance, alpha=0.05, method='bonf')
    # reject hypothesis that mean is the same? rej=true means a sign. difference exists

    '''
    `a` looks like this:

    Multiple Comparison of Means - Tukey HSD,FWER=0.01
    ===============================================
     group1  group2 meandiff  lower   upper  reject
    -----------------------------------------------
    57-APDT 58-APDT -0.0487  -0.0759 -0.0215  True
    59-APDT 60-APDT -0.0124  -0.0395  0.0148 False
    -----------------------------------------------
    '''
    data = str(a).split('\n')
    desc = data[0]
    header = data[2].split()
    header[-1] = 'significant'
    rows = [row.split() for row in data[4:-1]]

    return Table(header, rows, desc), sorted(set(names))


def pairwise_randomised_significance(exp1, exp2, clf=CLASSIFIER, nboot=500, statistic=accuracy_score):
    # https://stat.duke.edu/~ar182/rr/examples-gallery/PermutationTest.html
    # http://stackoverflow.com/a/24801874/419338
    y = Results.objects.get(id=exp1, classifier=clf).predictions
    y_gold = Results.objects.get(id=exp1, classifier=clf).gold
    y_size = len(y)
    z = Results.objects.get(id=exp2, classifier=clf).predictions
    z_gold = Results.objects.get(id=exp2, classifier=clf).gold

    # todo these will not be the same size as different thesauri may drop different documents from
    # the test set. shuffling code below needs to be updated
    # assert len(y) == len(z) == len(gold)
    # assert list(gold) == list(Results.objects.get(id=exp2, classifier=clf).gold)

    # at least check these experiments were done on the same dataset
    assert set(y) == set(z) == set(y_gold) == set(z_gold)

    y_acc = statistic(y_gold, y)
    z_acc = statistic(z_gold, z)
    theta_hat = np.abs(y_acc - z_acc)
    print('Original difference', theta_hat)

    estimates = []
    pooled = np.hstack([y, z]).ravel()
    pooled_g = np.hstack([y_gold, z_gold]).ravel()
    for _ in range(nboot):
        perm_ind = np.arange(len(pooled))
        np.random.shuffle(perm_ind)
        pooled = pooled[perm_ind]
        pooled_g = pooled_g[perm_ind]

        y_acc = statistic(pooled_g[:y_size], pooled[:y_size])
        z_acc = statistic(pooled_g[y_size:], pooled[y_size:])
        new_diff = np.abs(y_acc - z_acc)
        # print('New difference', new_diff)
        estimates.append(new_diff)

    diff_count = len(np.where(estimates <= theta_hat)[0])
    hat_asl_perm = 1.0 - (diff_count / nboot)
    return hat_asl_perm


def get_composer_name(n):
    vectors = Experiment.objects.get(id=n).vectors
    return vectors.composer if vectors else None
