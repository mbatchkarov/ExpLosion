from io import BytesIO
import base64
import re

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import MultiComparison
import statsmodels.api as sm
from critical_difference.plot import do_plot, print_figure

from gui.models import Experiment, Table, get_results_table

CLASSIFIER = 'MultinomialNB'
METRIC = 'accuracy_score'


def populate_manually():
    # run manually in django console to populate the database

    with open('experiments.txt') as infile:
        table_descr = [eval(x)[0] for x in infile.readlines()]
    # '5,gigaw,R2,0,Add,AN_NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler' ,
    # '136,word2vec,MR,100,Right,AN_NN,word2vec,0,1,0,SignifiedOnlyFeatureHandler',
    for line in table_descr:
        print(line)
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


class BaseExplosionAnalysis(object):
    @staticmethod
    def populate_experiments_db(*args, **kwargs):
        if not Experiment.objects.count():
            Experiment.objects.all().delete()
            populate_manually()

    @staticmethod
    def get_tables(exp_ids):
        return [
            Table(['num', 'val', 'val', 'val'],
                  [[1, 'test', 1, 1], [2, 'test', 2, 2]],
                  'some table'),
            Table(['num', 'val', 'val', 'val'],
                  [[1, 'test', 1, 1], [2, 'test', 2, 2]],
                  'some other table')
        ]

    @staticmethod
    def get_generated_figures(exp_ids):
        base64_images = []
        for _ in range(2):
            fig = plt.Figure(dpi=100, facecolor='white')
            ax = fig.add_subplot(111)

            # todo do whatever magic is needed here
            ax.plot(range(10))

            canvas = FigureCanvas(fig)
            s = BytesIO()
            canvas.print_png(s)
            base64_images.append(base64.b64encode(s.getvalue()))
        return base64_images

    @staticmethod
    def get_static_figures(exp_ids):
        return ['static/img/test.jpg']


class Thesisgen(BaseExplosionAnalysis):
    @staticmethod
    def get_tables(exp_ids):
        return [
            Thesisgen.get_performance_table(exp_ids),
            Thesisgen.get_significance_table(exp_ids)[0],
        ] if exp_ids else []

    @staticmethod
    def get_static_figures(exp_ids):
        # return [
        # "static/figures/stats-exp%d-0.png" % n for n in exp_ids
        # ] if exp_ids else []
        return []

    @staticmethod
    def get_generated_figures(exp_ids):
        return [Thesisgen.get_demsar_diagram(exp_ids)] if exp_ids else []  # Thesisgen.get_r2_correlation_plot(exp_ids)

    @staticmethod
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
                composer_name = Thesisgen.get_composer_name(n)
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def get_r2_correlation_plot(exp_ids):
        print('running r2 scatter')

        logs = []
        for n in exp_ids:
            with open('gui/static/figures/stats_output%d.txt' % n) as infile:
                logs.append(''.join(infile.readlines()))

        # selected_r2 = Thesisgen._get_r2_from_log(exp_ids, logs)
        # selected_sse = Thesisgen._get_SSE_from_log(exp_ids, logs)
        selected_acc = []
        acc_err = []
        for n in exp_ids:
            sample_size, score_mean, score_std = get_results_table(n).objects.all(). \
                filter(classifier=CLASSIFIER, metric=METRIC)[0].get_performance_info()
            selected_acc.append(score_mean)
            acc_err.append(score_std)

            # return \
            # Thesisgen._plot_x_agains_accuracy(selected_sse,
            # selected_acc,
            # acc_err,
            # exp_ids,
            # title='Normalised SSE from diagonal'), \
            # Thesisgen._plot_x_agains_accuracy(selected_r2,
            # selected_acc,
            # acc_err,
            # exp_ids,
            # title='R2 of good feature LOR scatter plot'), \
        return Thesisgen._plot_x_agains_accuracy(Thesisgen._get_wrong_quadrant_pct(exp_ids, logs, 'unweighted'),
                                                 selected_acc,
                                                 acc_err,
                                                 exp_ids,
                                                 title='Pct in wrong quadrant (unweighted)'), \
               Thesisgen._plot_x_agains_accuracy(Thesisgen._get_wrong_quadrant_pct(exp_ids, logs, 'weighted by freq'),
                                                 selected_acc,
                                                 acc_err,
                                                 exp_ids,
                                                 title='Pct in wrong quadrant (weighted by freq)'), \
               Thesisgen._plot_x_agains_accuracy(Thesisgen._get_wrong_quadrant_pct(exp_ids, logs, 'weighted by sim'),
                                                 selected_acc,
                                                 acc_err,
                                                 exp_ids,
                                                 title='Pct in wrong quadrant (weighted by sim)'), \
               Thesisgen._plot_x_agains_accuracy(Thesisgen._get_wrong_quadrant_pct(exp_ids,
                                                                                   logs, 'weighted by sim and freq'),
                                                 selected_acc,
                                                 acc_err,
                                                 exp_ids,
                                                 title='Pct in wrong quadrant (weighted by sim and freq)'), \
            # Thesisgen._plot_x_agains_accuracy(Thesisgen._get_wrong_quadrant_pct(exp_ids, logs,
        # 'weighted by seriousness, '
        # 'sim and freq'),
        # selected_acc,
        # acc_err,
        # exp_ids,
        # title='Pct in wrong quadrant (weighted by seriousness, sim and freq)')

    @staticmethod
    def figure_to_base64(fig):
        canvas = FigureCanvas(fig)
        s = BytesIO()
        canvas.print_png(s)
        return base64.b64encode(s.getvalue())

    @staticmethod
    def _plot_x_agains_accuracy(x, selected_acc, acc_err, exp_ids, title=''):
        fig = plt.Figure(dpi=100, facecolor='white')
        ax = fig.add_subplot(111)

        # do whatever magic is needed here
        coef, r2, r2adj = Thesisgen.plot_regression_line(ax,
                                                         np.array(x),
                                                         np.array(selected_acc),
                                                         np.ones(len(selected_acc)))
        ax.errorbar(x, selected_acc, yerr=acc_err, capsize=0, ls='none')
        composer_names = []
        for n in exp_ids:
            composer_name = Thesisgen.get_composer_name(n)
            composer_names.append('%d-%s' % (n, composer_name))
        ax.scatter(x, selected_acc)
        for i, txt in enumerate(composer_names):
            ax.annotate(txt, (x[i], selected_acc[i]), fontsize='xx-small', rotation=30)

        if len(coef) > 1:
            fig.suptitle('y=%.2fx%+.2f; r2=%.2f(%.2f)' % (coef[0], coef[1], r2, r2adj))
        else:
            fig.suptitle('All x values are 0, cannot fit regression line')
        ax.set_xlabel(title)
        ax.set_ylabel(METRIC)
        ax.axhline(y=0.5, linestyle='--', color='0.75')
        return Thesisgen.figure_to_base64(fig)


    @staticmethod
    def make_df(table:Table):
        df = pd.DataFrame(table.rows, columns=table.header)
        # df.set_index('composer')
        return df

    @staticmethod
    def get_demsar_diagram(exp_ids):
        table = Thesisgen.get_performance_table(exp_ids)
        sign_table, names = Thesisgen.get_significance_table(exp_ids)
        df = Thesisgen.make_df(sign_table)
        names = np.array(['%d-%s' % (x[0], x[2]) for x in table.rows])
        scores = np.array([float(x[4][:-1]) for x in table.rows])

        idx = np.argsort(scores)
        scores = list(scores[idx])
        names = list(names[idx])

        # print(list(df.group1), list(df.group2), list(df.significant))
        def get_insignificant_pairs(*args):
            mylist = []
            for a, b, significant_diff in zip(df.group1, df.group2, df.significant):
                if significant_diff == 'False':  # convert str
                    mylist.append((names.index(a), names.index(b)))
            return sorted(set(tuple(sorted(x)) for x in mylist))

        fig = do_plot(scores, get_insignificant_pairs, names)
        print_figure(fig, "%s.png" % ('_'.join(sorted(names))), format='png')

        return Thesisgen.figure_to_base64(fig)

    @staticmethod
    def get_performance_table(exp_ids):
        print('running performance query')
        data = []
        sample_size = 500
        for n in exp_ids:
            composer_name = Thesisgen.get_composer_name(n)
            results = get_results_table(n).objects.all().filter(metric=METRIC,
                                                                classifier=CLASSIFIER,
                                                                sample_size=sample_size)
            if not results.exists():
                # table or result does not exist
                print('skipping table %d and classifier %s' % (n, CLASSIFIER))
                continue

            size, acc, acc_stderr = results[0].get_performance_info()
            data.append([n, CLASSIFIER, composer_name,
                         sample_size, '{:.2%}'.format(acc), '{:.2%}'.format(acc_stderr)])

        table = Table(['id', 'classifier', 'composer', 'sample size', METRIC, 'std error'],
                      data,
                      'Performance at 500 training documents')
        return table

    @staticmethod
    def get_significance_table(exp_ids, classifier='MultinomialNB'):
        # get human-readable labels for the table
        data = []
        composers = []

        print('Running significance for experiments %r' % exp_ids)
        for n in exp_ids:
            # human-readable name
            composer_name = Thesisgen.get_composer_name(n)
            cv_folds = get_results_table(n).objects.values_list('cv_folds', flat=True)[0]
            composers.extend(['%d-%s' % (n, composer_name)] * cv_folds)

            # get scores for each CV run- these aren't in the database
            # only at size 500
            outfile = '../thesisgenerator/conf/exp{0}/output/exp{0}-0.out-raw.csv'.format(n)
            df = pd.read_csv(outfile)
            mask = df['classifier'].isin([classifier]) & df['metric'].isin([METRIC])
            ordered_scores = df['score'][mask].tolist()
            data.append(ordered_scores)

            # #plot distribution of scores to visually check for normality
            # plt.figure()
            # plt.hist(sorted(ordered_scores), bins=12)
            # plt.savefig('distrib%d.png' % n, format='png')
        data = np.hstack(data)
        mod = MultiComparison(data, composers, group_order=sorted(set(composers)))
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

    @staticmethod
    def plot_regression_line(ax, x, y, weights):
        # copied here from thesisgenerator.scripts.plot
        xs = np.linspace(min(x), max(x))
        x1 = sm.add_constant(x)
        model = sm.WLS(y, x1, weights=weights)
        results = model.fit()
        coef = results.params[::-1]  # statsmodels' linear equation is b+ax, numpy's is ax+b
        ax.plot(xs, results.predict(sm.add_constant(xs)), 'r-')
        return coef, results.rsquared, results.rsquared_adj

    @staticmethod
    def get_composer_name(n):
        composer_name = Experiment.objects.values_list('composer', flat=True).filter(id=n)[0]
        return composer_name
