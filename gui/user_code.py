from io import BytesIO
import base64
import re

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import MultiComparison
import statsmodels.api as sm

from gui.models import Experiment, Table, get_results_table

CLASSIFIER = 'MultinomialNB'
METRIC = 'accuracy_score'


def populate_manually():
    # run manually in django console to populate the database
    table_descr = [
        '1,-,R2,-1,Random,AN_NN,-,0,1,1,SignifiedOnlyFeatureHandler',
        '2,-,R2,-1,Signifier,AN_NN,-,0,1,0,BaseFeatureHandler',
        '3,-,MR,-1,Random,AN_NN,-,0,1,1,SignifiedOnlyFeatureHandler',
        '4,-,MR,-1,Signifier,AN_NN,-,0,1,0,BaseFeatureHandler',
        '5,gigaw,R2,0,Add,AN_NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '6,gigaw,R2,0,Mult,AN_NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '7,gigaw,R2,0,Left,AN_NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '8,gigaw,R2,0,Right,AN_NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '9,gigaw,R2,0,Observed,AN_NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '10,gigaw,R2,0,APDT,AN_NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '11,gigaw,R2,100,Add,AN_NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '12,gigaw,R2,100,Mult,AN_NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '13,gigaw,R2,100,Left,AN_NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '14,gigaw,R2,100,Right,AN_NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '15,gigaw,R2,100,Baroni,AN_NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '16,gigaw,R2,100,Observed,AN_NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '17,gigaw,R2,100,APDT,AN_NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '18,gigaw,MR,0,Add,AN_NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '19,gigaw,MR,0,Mult,AN_NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '20,gigaw,MR,0,Left,AN_NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '21,gigaw,MR,0,Right,AN_NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '22,gigaw,MR,0,Observed,AN_NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '23,gigaw,MR,0,APDT,AN_NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '24,gigaw,MR,100,Add,AN_NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '25,gigaw,MR,100,Mult,AN_NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '26,gigaw,MR,100,Left,AN_NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '27,gigaw,MR,100,Right,AN_NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '28,gigaw,MR,100,Baroni,AN_NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '29,gigaw,MR,100,Observed,AN_NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '30,gigaw,MR,100,APDT,AN_NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '31,gigaw,R2,0,Add,AN_NN,windows,0,1,0,SignifiedOnlyFeatureHandler',
        '32,gigaw,R2,0,Mult,AN_NN,windows,0,1,0,SignifiedOnlyFeatureHandler',
        '33,gigaw,R2,0,Left,AN_NN,windows,0,1,0,SignifiedOnlyFeatureHandler',
        '34,gigaw,R2,0,Right,AN_NN,windows,0,1,0,SignifiedOnlyFeatureHandler',
        '35,gigaw,R2,0,Observed,AN_NN,windows,0,1,0,SignifiedOnlyFeatureHandler',
        '36,gigaw,R2,100,Add,AN_NN,windows,0,1,0,SignifiedOnlyFeatureHandler',
        '37,gigaw,R2,100,Mult,AN_NN,windows,0,1,0,SignifiedOnlyFeatureHandler',
        '38,gigaw,R2,100,Left,AN_NN,windows,0,1,0,SignifiedOnlyFeatureHandler',
        '39,gigaw,R2,100,Right,AN_NN,windows,0,1,0,SignifiedOnlyFeatureHandler',
        '40,gigaw,R2,100,Baroni,AN_NN,windows,0,1,0,SignifiedOnlyFeatureHandler',
        '41,gigaw,R2,100,Observed,AN_NN,windows,0,1,0,SignifiedOnlyFeatureHandler',
        '42,gigaw,MR,0,Add,AN_NN,windows,0,1,0,SignifiedOnlyFeatureHandler',
        '43,gigaw,MR,0,Mult,AN_NN,windows,0,1,0,SignifiedOnlyFeatureHandler',
        '44,gigaw,MR,0,Left,AN_NN,windows,0,1,0,SignifiedOnlyFeatureHandler',
        '45,gigaw,MR,0,Right,AN_NN,windows,0,1,0,SignifiedOnlyFeatureHandler',
        '46,gigaw,MR,0,Observed,AN_NN,windows,0,1,0,SignifiedOnlyFeatureHandler',
        '47,gigaw,MR,100,Add,AN_NN,windows,0,1,0,SignifiedOnlyFeatureHandler',
        '48,gigaw,MR,100,Mult,AN_NN,windows,0,1,0,SignifiedOnlyFeatureHandler',
        '49,gigaw,MR,100,Left,AN_NN,windows,0,1,0,SignifiedOnlyFeatureHandler',
        '50,gigaw,MR,100,Right,AN_NN,windows,0,1,0,SignifiedOnlyFeatureHandler',
        '51,gigaw,MR,100,Baroni,AN_NN,windows,0,1,0,SignifiedOnlyFeatureHandler',
        '52,gigaw,MR,100,Observed,AN_NN,windows,0,1,0,SignifiedOnlyFeatureHandler',
        '53,neuro,R2,100,Socher,AN_NN,neuro,0,1,0,SignifiedOnlyFeatureHandler',
        '54,neuro,MR,100,Socher,AN_NN,neuro,0,1,0,SignifiedOnlyFeatureHandler',
        '55,gigaw,R2,0,Add,AN_NN,dependencies,0,0,0,SignifiedOnlyFeatureHandler',
        '56,gigaw,R2,0,Mult,AN_NN,dependencies,0,0,0,SignifiedOnlyFeatureHandler',
        '57,gigaw,R2,0,Left,AN_NN,dependencies,0,0,0,SignifiedOnlyFeatureHandler',
        '58,gigaw,R2,0,Right,AN_NN,dependencies,0,0,0,SignifiedOnlyFeatureHandler',
        '59,gigaw,R2,0,Observed,AN_NN,dependencies,0,0,0,SignifiedOnlyFeatureHandler',
        '60,gigaw,R2,0,APDT,AN_NN,dependencies,0,0,0,SignifiedOnlyFeatureHandler',
        '61,gigaw,R2,100,Add,AN_NN,dependencies,0,0,0,SignifiedOnlyFeatureHandler',
        '62,gigaw,R2,100,Mult,AN_NN,dependencies,0,0,0,SignifiedOnlyFeatureHandler',
        '63,gigaw,R2,100,Left,AN_NN,dependencies,0,0,0,SignifiedOnlyFeatureHandler',
        '64,gigaw,R2,100,Right,AN_NN,dependencies,0,0,0,SignifiedOnlyFeatureHandler',
        '65,gigaw,R2,100,Baroni,AN_NN,dependencies,0,0,0,SignifiedOnlyFeatureHandler',
        '66,gigaw,R2,100,Observed,AN_NN,dependencies,0,0,0,SignifiedOnlyFeatureHandler',
        '67,gigaw,R2,100,APDT,AN_NN,dependencies,0,0,0,SignifiedOnlyFeatureHandler',
        '68,gigaw,MR,0,Add,AN_NN,dependencies,0,0,0,SignifiedOnlyFeatureHandler',
        '69,gigaw,MR,0,Mult,AN_NN,dependencies,0,0,0,SignifiedOnlyFeatureHandler',
        '70,gigaw,MR,0,Left,AN_NN,dependencies,0,0,0,SignifiedOnlyFeatureHandler',
        '71,gigaw,MR,0,Right,AN_NN,dependencies,0,0,0,SignifiedOnlyFeatureHandler',
        '72,gigaw,MR,0,Observed,AN_NN,dependencies,0,0,0,SignifiedOnlyFeatureHandler',
        '73,gigaw,MR,0,APDT,AN_NN,dependencies,0,0,0,SignifiedOnlyFeatureHandler',
        '74,gigaw,MR,100,Add,AN_NN,dependencies,0,0,0,SignifiedOnlyFeatureHandler',
        '75,gigaw,MR,100,Mult,AN_NN,dependencies,0,0,0,SignifiedOnlyFeatureHandler',
        '76,gigaw,MR,100,Left,AN_NN,dependencies,0,0,0,SignifiedOnlyFeatureHandler',
        '77,gigaw,MR,100,Right,AN_NN,dependencies,0,0,0,SignifiedOnlyFeatureHandler',
        '78,gigaw,MR,100,Baroni,AN_NN,dependencies,0,0,0,SignifiedOnlyFeatureHandler',
        '79,gigaw,MR,100,Observed,AN_NN,dependencies,0,0,0,SignifiedOnlyFeatureHandler',
        '80,gigaw,MR,100,APDT,AN_NN,dependencies,0,0,0,SignifiedOnlyFeatureHandler',
        '81,gigaw,R2,0,Add,AN_NN,windows,0,0,0,SignifiedOnlyFeatureHandler',
        '82,gigaw,R2,0,Mult,AN_NN,windows,0,0,0,SignifiedOnlyFeatureHandler',
        '83,gigaw,R2,0,Left,AN_NN,windows,0,0,0,SignifiedOnlyFeatureHandler',
        '84,gigaw,R2,0,Right,AN_NN,windows,0,0,0,SignifiedOnlyFeatureHandler',
        '85,gigaw,R2,0,Observed,AN_NN,windows,0,0,0,SignifiedOnlyFeatureHandler',
        '86,gigaw,R2,100,Add,AN_NN,windows,0,0,0,SignifiedOnlyFeatureHandler',
        '87,gigaw,R2,100,Mult,AN_NN,windows,0,0,0,SignifiedOnlyFeatureHandler',
        '88,gigaw,R2,100,Left,AN_NN,windows,0,0,0,SignifiedOnlyFeatureHandler',
        '89,gigaw,R2,100,Right,AN_NN,windows,0,0,0,SignifiedOnlyFeatureHandler',
        '90,gigaw,R2,100,Baroni,AN_NN,windows,0,0,0,SignifiedOnlyFeatureHandler',
        '91,gigaw,R2,100,Observed,AN_NN,windows,0,0,0,SignifiedOnlyFeatureHandler',
        '92,gigaw,MR,0,Add,AN_NN,windows,0,0,0,SignifiedOnlyFeatureHandler',
        '93,gigaw,MR,0,Mult,AN_NN,windows,0,0,0,SignifiedOnlyFeatureHandler',
        '94,gigaw,MR,0,Left,AN_NN,windows,0,0,0,SignifiedOnlyFeatureHandler',
        '95,gigaw,MR,0,Right,AN_NN,windows,0,0,0,SignifiedOnlyFeatureHandler',
        '96,gigaw,MR,0,Observed,AN_NN,windows,0,0,0,SignifiedOnlyFeatureHandler',
        '97,gigaw,MR,100,Add,AN_NN,windows,0,0,0,SignifiedOnlyFeatureHandler',
        '98,gigaw,MR,100,Mult,AN_NN,windows,0,0,0,SignifiedOnlyFeatureHandler',
        '99,gigaw,MR,100,Left,AN_NN,windows,0,0,0,SignifiedOnlyFeatureHandler',
        '100,gigaw,MR,100,Right,AN_NN,windows,0,0,0,SignifiedOnlyFeatureHandler',
        '101,gigaw,MR,100,Baroni,AN_NN,windows,0,0,0,SignifiedOnlyFeatureHandler',
        '102,gigaw,MR,100,Observed,AN_NN,windows,0,0,0,SignifiedOnlyFeatureHandler',
        '103,neuro,R2,100,Socher,AN_NN,neuro,0,0,0,SignifiedOnlyFeatureHandler',
        '104,neuro,MR,100,Socher,AN_NN,neuro,0,0,0,SignifiedOnlyFeatureHandler',
        '105,gigaw,R2,100,Add,AN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '106,gigaw,R2,100,Mult,AN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '107,gigaw,R2,100,Left,AN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '108,gigaw,R2,100,Right,AN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '109,gigaw,R2,100,Baroni,AN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '110,gigaw,R2,100,Observed,AN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '111,gigaw,R2,100,APDT,AN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '112,neuro,R2,100,Socher,AN,neuro,0,1,0,SignifiedOnlyFeatureHandler',
        '113,gigaw,R2,100,Add,NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '114,gigaw,R2,100,Mult,NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '115,gigaw,R2,100,Left,NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '116,gigaw,R2,100,Right,NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '117,gigaw,R2,100,Baroni,NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '118,gigaw,R2,100,Observed,NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '119,gigaw,R2,100,APDT,NN,dependencies,0,1,0,SignifiedOnlyFeatureHandler',
        '120,neuro,R2,100,Socher,NN,neuro,0,1,0,SignifiedOnlyFeatureHandler',
        '121,neuro,R2,100,Add,AN_NN,neuro,0,1,0,SignifiedOnlyFeatureHandler',
        '122,neuro,R2,100,Mult,AN_NN,neuro,0,1,0,SignifiedOnlyFeatureHandler',
        '123,neuro,R2,100,Left,AN_NN,neuro,0,1,0,SignifiedOnlyFeatureHandler',
        '124,neuro,R2,100,Right,AN_NN,neuro,0,1,0,SignifiedOnlyFeatureHandler',
        '125,neuro,MR,100,Add,AN_NN,neuro,0,1,0,SignifiedOnlyFeatureHandler',
        '126,neuro,MR,100,Mult,AN_NN,neuro,0,1,0,SignifiedOnlyFeatureHandler',
        '127,neuro,MR,100,Left,AN_NN,neuro,0,1,0,SignifiedOnlyFeatureHandler',
        '128,neuro,MR,100,Right,AN_NN,neuro,0,1,0,SignifiedOnlyFeatureHandler',
        '129,word2vec,R2,100,Add,AN_NN,word2vec,0,1,0,SignifiedOnlyFeatureHandler',
        '130,word2vec,R2,100,Mult,AN_NN,word2vec,0,1,0,SignifiedOnlyFeatureHandler',
        '131,word2vec,R2,100,Left,AN_NN,word2vec,0,1,0,SignifiedOnlyFeatureHandler',
        '132,word2vec,R2,100,Right,AN_NN,word2vec,0,1,0,SignifiedOnlyFeatureHandler',
        '133,word2vec,MR,100,Add,AN_NN,word2vec,0,1,0,SignifiedOnlyFeatureHandler',
        '134,word2vec,MR,100,Mult,AN_NN,word2vec,0,1,0,SignifiedOnlyFeatureHandler',
        '135,word2vec,MR,100,Left,AN_NN,word2vec,0,1,0,SignifiedOnlyFeatureHandler',
        '136,word2vec,MR,100,Right,AN_NN,word2vec,0,1,0,SignifiedOnlyFeatureHandler',
    ]
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
            # ThesisgeneratorExplosionAnalysis.get_significance_table(exp_ids),
        ] if exp_ids else []

    @staticmethod
    def get_static_figures(exp_ids):
        return [
            "static/figures/stats-exp%d-0.png" % n for n in exp_ids
        ] if exp_ids else []

    @staticmethod
    def get_generated_figures(exp_ids):
        return Thesisgen.get_r2_correlation_plot(exp_ids) if exp_ids else []

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
            #                                   acc_err,
            #                                   exp_ids,
            #                                   title='Normalised SSE from diagonal'), \
            # Thesisgen._plot_x_agains_accuracy(selected_r2,
            #                                   selected_acc,
            #                                   acc_err,
            #                                   exp_ids,
            #                                   title='R2 of good feature LOR scatter plot'), \
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
        #                                                                     'sim and freq'),
        #                                   selected_acc,
        #                                   acc_err,
        #                                   exp_ids,
        #                                   title='Pct in wrong quadrant (weighted by seriousness, sim and freq)')

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

        canvas = FigureCanvas(fig)
        s = BytesIO()
        canvas.print_png(s)
        return base64.b64encode(s.getvalue())

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

        return Table(['id', 'classifier', 'composer', 'sample size', METRIC, 'std error'],
                     data,
                     'Performance at 500 training documents')

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
        57-APDT 59-APDT -0.2581  -0.2853 -0.2309  True
        57-APDT 60-APDT -0.2705  -0.2977 -0.2433  True
        58-APDT 59-APDT -0.2094  -0.2366 -0.1823  True
        58-APDT 60-APDT -0.2218   -0.249 -0.1946  True
        59-APDT 60-APDT -0.0124  -0.0395  0.0148 False
        -----------------------------------------------
        '''
        data = str(a).split('\n')
        desc = data[0]
        header = data[2].split()
        rows = [row.split() for row in data[4:-1]]

        return Table(header, rows, desc)

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
