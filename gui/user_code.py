from io import BytesIO
import base64
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import MultiComparison
from critical_difference.plot import do_plot, print_figure
from gui.constants import METRIC_DB, CLASSIFIER, ABBREVIATIONS, BOOTSTRAP_REPS
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

    names = pretty_names(exp_ids, name_format)
    full_names = []
    for eid, this_name in zip(exp_ids, names):
        document_labels = get_data_for_signif_test(eid)
        full_names.extend([this_name] * len(document_labels))

    data = np.concatenate([get_data_for_signif_test(i) for i in exp_ids])
    sign_table, _ = get_significance_table(exp_ids, data=data, names=full_names)

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
        raise ValueError('Oh shite')

    if len(exp_ids) < 2:
        print('Cannot run significance test on less than 2 methods: %r' % set(names))
        return None, None

    mod = MultiComparison(np.array(data), names, group_order=sorted(set(names)))
    # a = mod.tukeyhsd(alpha=0.05)
    res_table, cock, balls = mod.allpairtest(pairwise_randomised_significance, alpha=0.05, method='bonf', pvalidx=0)
    # reject hypothesis that mean is the same? rej=true means a sign. difference exists
    # print(str(res_table))
    '''
    `a` looks like this:

    Test Multiple Comparison pairwise_randomised_significance
    FWER=0.05 method=bonf
    alphacSidak=0.05, alphacBonf=0.050
    ===================================================
       group1      group2   stat  pval pval_corr reject
    ---------------------------------------------------
    RandN-RandN RandV-RandV -1.0 0.184    -1.0    True
    ---------------------------------------------------
    '''
    data = str(res_table).split('\n')
    desc = ','.join(data[:3])
    header = data[4].split()
    header[-1] = 'significant'
    rows = [row.split() for row in data[6:-1]]

    return pd.DataFrame(rows, columns=header), sorted(set(names))


def get_data_for_signif_test(exp_id, clf=CLASSIFIER):
    y = Results.objects.get(id=exp_id, classifier=clf).predictions
    y_gold = Results.objects.get(id=exp_id, classifier=clf).gold

    # check gold standard matches right answers
    assert set(y) == set(y_gold)
    return np.concatenate([y, y_gold])


@memory.cache
def pairwise_randomised_significance(y, z, nboot=BOOTSTRAP_REPS, statistic=accuracy_score):
    # https://stat.duke.edu/~ar182/rr/examples-gallery/PermutationTest.html
    # http://stackoverflow.com/a/24801874/419338

    half_y = len(y) / 2
    y_gold = y[half_y:]
    y = y[:half_y]

    half_z = len(z) / 2
    z_gold = z[half_z:]
    z = z[:half_z]

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

        y_acc = statistic(pooled_g[:half_y], pooled[:half_y])
        z_acc = statistic(pooled_g[half_y:], pooled[half_y:])
        new_diff = np.abs(y_acc - z_acc)
        # print('New difference', new_diff)
        estimates.append(new_diff)
    diff_count = len(np.where(estimates <= theta_hat)[0])
    hat_asl_perm = 1.0 - (diff_count / nboot)
    print('p-value', hat_asl_perm)
    return theta_hat, hat_asl_perm
    # statsmodels multiple comparison insists this is shaped like this: (statistic, p-value)


def get_composer_name(n):
    vectors = Experiment.objects.get(id=n).expansions.vectors
    return vectors.composer if vectors else None
