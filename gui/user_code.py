from io import BytesIO
import base64
import re
from configobj import ConfigObj

import matplotlib as mpl
from thesisgenerator.utils.output_utils import get_scores, get_vectors_field

mpl.use('Agg')  # for running on headless servers

import validate
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import MultiComparison
import statsmodels.api as sm
from critical_difference.plot import do_plot, print_figure
from gui.models import Experiment, Table, Results

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
        res.append(get_performance_table(exp_ids))
    if config['significance_table']:
        res.append(get_significance_table(exp_ids)[0])
    return res


def get_static_figures(exp_ids):
    if config['static_figures'] and exp_ids:
        return ["static/figures/stats-exp%d-0.png" % n for n in exp_ids]
    else:
        return []


def get_generated_figures(exp_ids):
    if not exp_ids:
        return []
    res = []
    if config['r2_correlation']:
        res.append(get_r2_correlation_plot(exp_ids))
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


def _get_r2_from_log(exp_ids, logs):
    selected_r2 = []  # todo put this in database in advance?
    failed_experiments = set()
    for txt, n in zip(logs, exp_ids):
        try:
            # todo there's a better way to get this
            r2 = float(re.search('R-squared of class-pull plot: ([0-9\.]+)', txt).groups()[0])
        except AttributeError:
            # groups() fails if there isn't a match. This happens when the detailed offline analysis
            # failed for some reason. Let's try and recover
            composer_name = get_composer_name(n)
            if composer_name == 'Random':
                # todo If f(x) is random, there shouldn't be any correlation between x and f(x). In this case,
                # for a given feature x and its log likelihood ratio, f(x) is the LLR of some other random feature.
                r2 = 0
            elif composer_name == 'Signifier':
                # todo If f(x) = x, there shouldn't be perfect correlation between x and f(x). In this case,
                # the r2 metric should be 1.
                r2 = 1
            else:
                # raise ValueError('Detailed analysis of experiment %d failed, check output directory' % n)
                print('Detailed analysis of experiment %d failed, check output directory' % n)
                r2 = 0
                # failed_experiments.add(n)
                # continue
                # raise ValueError('Detailed analysis of experiment %d failed, check output directory' % n)
        selected_r2.append(r2)
    return selected_r2


def _get_SSE_from_log(exp_ids, logs):
    selected_scores = []  # todo put this in database in advance?
    for txt, n in zip(logs, exp_ids):
        try:
            score = float(
                re.search('Sum-of-squares error compared to perfect diagonal = ([0-9\.]+)', txt).groups()[0])
        except AttributeError:
            score = 0
        selected_scores.append(score)
    return selected_scores


def _get_wrong_quadrant_pct(exp_ids, logs, weighted):
    selected_scores = []
    for txt, n in zip(logs, exp_ids):
        try:
            score = re.search('([0-9/\.]+) data points are in the wrong quadrant \(%s\)' % weighted, txt).groups()[
                0]
            score = score.split('/')
            selected_scores.append(float(score[0]) / float(score[1]))
        except AttributeError:
            selected_scores.append(0)
    return selected_scores


def get_r2_correlation_plot(exp_ids):
    print('running r2 scatter')

    logs = []
    for n in exp_ids:
        with open('gui/static/figures/stats_output%d.txt' % n) as infile:
            logs.append(''.join(infile.readlines()))

    # selected_r2 = _get_r2_from_log(exp_ids, logs)
    # selected_sse = _get_SSE_from_log(exp_ids, logs)
    selected_acc = []
    acc_err = []
    for n in exp_ids:
        sample_size, score_mean, score_std = Results.get(id=n, classifier=CLASSIFIER).get_performance_info(METRIC_DB)
        selected_acc.append(score_mean)
        acc_err.append(score_std)

        # return \
        # _plot_x_agains_accuracy(selected_sse,
        # selected_acc,
        # acc_err,
        # exp_ids,
        # title='Normalised SSE from diagonal'), \
        # _plot_x_agains_accuracy(selected_r2,
        # selected_acc,
        # acc_err,
        # exp_ids,
        # title='R2 of good feature LOR scatter plot'), \
    return _plot_x_agains_accuracy(_get_wrong_quadrant_pct(exp_ids, logs, 'unweighted'),
                                   selected_acc,
                                   acc_err,
                                   exp_ids,
                                   title='Pct in wrong quadrant (unweighted)'), \
           _plot_x_agains_accuracy(_get_wrong_quadrant_pct(exp_ids, logs, 'weighted by freq'),
                                   selected_acc,
                                   acc_err,
                                   exp_ids,
                                   title='Pct in wrong quadrant (weighted by freq)'), \
           _plot_x_agains_accuracy(_get_wrong_quadrant_pct(exp_ids, logs, 'weighted by sim'),
                                   selected_acc,
                                   acc_err,
                                   exp_ids,
                                   title='Pct in wrong quadrant (weighted by sim)'), \
           _plot_x_agains_accuracy(_get_wrong_quadrant_pct(exp_ids,
                                                           logs, 'weighted by sim and freq'),
                                   selected_acc,
                                   acc_err,
                                   exp_ids,
                                   title='Pct in wrong quadrant (weighted by sim and freq)'), \
        # _plot_x_agains_accuracy(_get_wrong_quadrant_pct(exp_ids, logs,
    # 'weighted by seriousness, '
    # 'sim and freq'),
    # selected_acc,
    # acc_err,
    # exp_ids,
    # title='Pct in wrong quadrant (weighted by seriousness, sim and freq)')


def figure_to_base64(fig):
    canvas = FigureCanvas(fig)
    s = BytesIO()
    canvas.print_png(s)
    return base64.b64encode(s.getvalue())


def _plot_x_agains_accuracy(x, selected_acc, acc_err, exp_ids, title=''):
    fig = plt.Figure(dpi=100, facecolor='white')
    ax = fig.add_subplot(111)

    # do whatever magic is needed here
    coef, r2, r2adj = plot_regression_line(ax,
                                           np.array(x),
                                           np.array(selected_acc),
                                           np.ones(len(selected_acc)))
    ax.errorbar(x, selected_acc, yerr=acc_err, capsize=0, ls='none')
    composer_names = []
    for n in exp_ids:
        composer_name = get_composer_name(n)
        composer_names.append('%d-%s' % (n, composer_name))
    ax.scatter(x, selected_acc)
    for i, txt in enumerate(composer_names):
        ax.annotate(txt, (x[i], selected_acc[i]), fontsize='xx-small', rotation=30)

    if len(coef) > 1:
        fig.suptitle('y=%.2fx%+.2f; r2=%.2f(%.2f)' % (coef[0], coef[1], r2, r2adj))
    else:
        fig.suptitle('All x values are 0, cannot fit regression line')
    ax.set_xlabel(title)
    ax.set_ylabel(METRIC_DB)
    ax.axhline(y=0.5, linestyle='--', color='0.75')
    return figure_to_base64(fig)


def make_df(table:Table, index_cols=None):
    df = pd.DataFrame(table.rows, columns=table.header)
    if index_cols:
        df.set_index(index_cols, inplace=True)
    return df


def get_demsar_params(exp_ids):
    scores_table = make_df(get_performance_table(exp_ids), 'exp id').convert_objects(convert_numeric=True)

    data, folds, exp_ids = get_scores(exp_ids)
    composers = ['%s-%s' % (a, b) for a, b in zip(get_vectors_field(exp_ids, 'algorithm'), \
                                                  get_vectors_field(exp_ids, 'composer'))]
    short_composers = ['%s-%s' % (a, b) for a, b in zip(get_vectors_field(exp_ids, 'algorithm', cv_folds=1), \
                                                        get_vectors_field(exp_ids, 'composer', cv_folds=1))]
    sign_table, _ = get_significance_table(exp_ids, data=data, composers=composers)
    sign_table = make_df(sign_table)

    return sign_table, np.array(short_composers), np.array(scores_table[METRIC_DB])


def get_demsar_diagram(significance_df, names, scores):
    idx = np.argsort(scores)
    scores = list(scores[idx])
    names = list(names[idx])
    print(significance_df)

    def get_insignificant_pairs(*args):
        mylist = []
        for a, b, significant_diff in zip(significance_df.group1,
                                          significance_df.group2,
                                          significance_df.significant):
            if significant_diff == 'False':
                mylist.append((names.index(a), names.index(b)))
        return sorted(set(tuple(sorted(x)) for x in mylist))

    # debug info
    from critical_difference.plot import merge_nonsignificant_cliques

    print(get_insignificant_pairs())
    print(merge_nonsignificant_cliques(get_insignificant_pairs()))

    fig = do_plot(scores, get_insignificant_pairs, names)
    fig.set_canvas(plt.gcf().canvas)
    filename = "sign-%s.png" % ('_'.join(sorted(set(names))))[:200] # there's a limit on that
    print('Saving figure to %s' % filename)
    print_figure(fig, filename, format='png')
    return fig


def get_performance_table(exp_ids):
    print('running performance query')

    # for exp_list in exp_lists:
    # if isinstance(exp_list, int):
    # exp_list = [exp_list]
    # composer = '%s-%s' % ('-'.join(str(foo) for foo in exp_list),
    # get_composer_name(exp_list[0]))
    # composers.extend([composer] * (cv_folds * len(exp_list)))
    # for exp_number in exp_list:

    all_data = []
    for exp_id in exp_ids:
        composer_name = '%s-%s' % (exp_id, get_composer_name(exp_id))
        results = Results.objects.filter(id=exp_id, classifier=CLASSIFIER)
        if not results:
            # table or result does not exist
            print('Missing results entry for exp %d and classifier %s' % (exp_id, CLASSIFIER))
            continue

        score_mean, score_std = results[0].get_performance_info(METRIC_DB)
        vectors = Experiment.objects.get(id=exp_id).vectors
        all_data.append([exp_id, str(vectors), CLASSIFIER, composer_name,
                         '{:.2}'.format(np.mean(score_mean)),
                         '{:.2}'.format(np.mean(score_std))])
    table = Table(['exp id', 'vectors', 'classifier', 'composer', METRIC_DB, 'std'],
                  all_data,
                  'Performance over crossvalidation (std is mean of [std_over_CV(exp) for exp in exp_id])')
    return table


# def _get_cv_scores_single_experiment(n, classifier):
# # human-readable name
# composer_name = get_composer_name(n)
#     # get scores for each CV run- these aren't in the database
#     # only at size 500
#     outfile = '../thesisgenerator/conf/exp{0}/output/exp{0}-0.out-raw.csv'.format(n)
#     if not os.path.exists(outfile):
#         print('Output file for exp %d missing' % n)
#         return None, '%d-%s' % (n, composer_name)
#     scores = pd.read_csv(outfile)
#     mask = (scores.classifier == classifier) & (scores.metric == METRIC_CSV_FILE)
#     ordered_scores = scores.score[mask].tolist()
#     return ordered_scores, '%d-%s' % (n, composer_name)
# def get_scores(exp_ids, classifier='MultinomialNB', cv_folds=25):
#     # get human-readable labels for the table
#     data = []
#     composers = []
#
#     for exp_number in exp_ids:
#         composer = '%d-%s' % (exp_number, get_composer_name(exp_number))
#         scores, _ = _get_cv_scores_single_experiment(exp_number, classifier)
#         if scores:
#             composers.extend([composer] * cv_folds)
#             data.extend(scores)
#     return data, composers


def get_significance_table(exp_ids, classifier='MultinomialNB', cv_folds=25, data=None, composers=None):
    print('Running significance for experiments %r' % exp_ids)
    if data is None and composers is None:
        data, composers, _ = get_scores(exp_ids, classifier=classifier, cv_folds=cv_folds)

    if len(set(composers)) < 2:
        raise ValueError('Cannot run significance test on less than 2 methods: %r' % set(composers))
    mod = MultiComparison(np.array(data), composers, group_order=sorted(set(composers)))
    a = mod.tukeyhsd(alpha=0.01)
    # reject hypothesis that mean is the same? rej=true means a sign. difference exists

    '''
    A looks like this:

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

    return Table(header, rows, desc), sorted(set(composers))


def plot_regression_line(ax, x, y, weights):
    # copied here from rator.scripts.plot
    xs = np.linspace(min(x), max(x))
    x1 = sm.add_constant(x)
    model = sm.WLS(y, x1, weights=weights)
    results = model.fit()
    coef = results.params[::-1]  # statsmodels' linear equation is b+ax, numpy's is ax+b
    ax.plot(xs, results.predict(sm.add_constant(xs)), 'r-')
    return coef, results.rsquared, results.rsquared_adj


def get_composer_name(n):
    vectors = Experiment.objects.get(id=n).vectors
    return vectors.composer if vectors else None
