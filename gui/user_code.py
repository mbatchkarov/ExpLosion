from cStringIO import StringIO
import base64
import re

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import MultiComparison
import statsmodels.api as sm

from gui.models import Experiment, Table, get_results_table


def populate_manually():
    # run manually in django console to populate the database
    table_descr = [
        '1,gigaw,R2,0,Add,AN_NN,dependencies,0',
        '2,gigaw,R2,0,Mult,AN_NN,dependencies,0',
        '3,gigaw,R2,0,Left,AN_NN,dependencies,0',
        '4,gigaw,R2,0,Right,AN_NN,dependencies,0',
        '5,gigaw,R2,0,Observed,AN_NN,dependencies,0',
        '6,gigaw,R2,100,Add,AN_NN,dependencies,0',
        '7,gigaw,R2,100,Mult,AN_NN,dependencies,0',
        '8,gigaw,R2,100,Left,AN_NN,dependencies,0',
        '9,gigaw,R2,100,Right,AN_NN,dependencies,0',
        '10,gigaw,R2,100,Baroni,AN_NN,dependencies,0',
        '11,gigaw,R2,100,Observed,AN_NN,dependencies,0',
        '12,gigaw,MR,0,Add,AN_NN,dependencies,0',
        '13,gigaw,MR,0,Mult,AN_NN,dependencies,0',
        '14,gigaw,MR,0,Left,AN_NN,dependencies,0',
        '15,gigaw,MR,0,Right,AN_NN,dependencies,0',
        '16,gigaw,MR,0,Observed,AN_NN,dependencies,0',
        '17,gigaw,MR,100,Add,AN_NN,dependencies,0',
        '18,gigaw,MR,100,Mult,AN_NN,dependencies,0',
        '19,gigaw,MR,100,Left,AN_NN,dependencies,0',
        '20,gigaw,MR,100,Right,AN_NN,dependencies,0',
        '21,gigaw,MR,100,Baroni,AN_NN,dependencies,0',
        '22,gigaw,MR,100,Observed,AN_NN,dependencies,0',
        '23,gigaw,R2,0,Add,AN_NN,windows,0',
        '24,gigaw,R2,0,Mult,AN_NN,windows,0',
        '25,gigaw,R2,0,Left,AN_NN,windows,0',
        '26,gigaw,R2,0,Right,AN_NN,windows,0',
        '27,gigaw,R2,0,Observed,AN_NN,windows,0',
        '28,gigaw,R2,100,Add,AN_NN,windows,0',
        '29,gigaw,R2,100,Mult,AN_NN,windows,0',
        '30,gigaw,R2,100,Left,AN_NN,windows,0',
        '31,gigaw,R2,100,Right,AN_NN,windows,0',
        '32,gigaw,R2,100,Baroni,AN_NN,windows,0',
        '33,gigaw,R2,100,Observed,AN_NN,windows,0',
        '34,gigaw,MR,0,Add,AN_NN,windows,0',
        '35,gigaw,MR,0,Mult,AN_NN,windows,0',
        '36,gigaw,MR,0,Left,AN_NN,windows,0',
        '37,gigaw,MR,0,Right,AN_NN,windows,0',
        '38,gigaw,MR,0,Observed,AN_NN,windows,0',
        '39,gigaw,MR,100,Add,AN_NN,windows,0',
        '40,gigaw,MR,100,Mult,AN_NN,windows,0',
        '41,gigaw,MR,100,Left,AN_NN,windows,0',
        '42,gigaw,MR,100,Right,AN_NN,windows,0',
        '43,gigaw,MR,100,Baroni,AN_NN,windows,0',
        '44,gigaw,MR,100,Observed,AN_NN,windows,0',
        '45,gigaw,R2,100,Add,AN,dependencies,0',
        '46,gigaw,R2,100,Mult,AN,dependencies,0',
        '47,gigaw,R2,100,Left,AN,dependencies,0',
        '48,gigaw,R2,100,Right,AN,dependencies,0',
        '49,gigaw,R2,100,Baroni,AN,dependencies,0',
        '50,gigaw,R2,100,Observed,AN,dependencies,0',
        '51,gigaw,R2,100,Add,NN,dependencies,0',
        '52,gigaw,R2,100,Mult,NN,dependencies,0',
        '53,gigaw,R2,100,Left,NN,dependencies,0',
        '54,gigaw,R2,100,Right,NN,dependencies,0',
        '55,gigaw,R2,100,Baroni,NN,dependencies,0',
        '56,gigaw,R2,100,Observed,NN,dependencies,0',
        '57,gigaw,R2,0,APDT,AN_NN,dependencies,0',
        '58,gigaw,R2,100,APDT,AN_NN,dependencies,0',
        '59,gigaw,MR,0,APDT,AN_NN,dependencies,0',
        '60,gigaw,MR,100,APDT,AN_NN,dependencies,0',
        '61,-,R2,100,Socher,AN_NN,-,0',
        '62,-,MR,100,Socher,AN_NN,-,0',
        '63,gigaw,R2,100,APDT,AN,dependencies,0',
        '64,-,R2,100,Socher,AN,-,0',
        '65,gigaw,MR,100,Add,AN,dependencies,0',
        '66,gigaw,MR,100,Mult,AN,dependencies,0',
        '67,gigaw,MR,100,Left,AN,dependencies,0',
        '68,gigaw,MR,100,Right,AN,dependencies,0',
        '69,gigaw,MR,100,Baroni,AN,dependencies,0',
        '70,gigaw,MR,100,Observed,AN,dependencies,0',
        '71,gigaw,MR,100,APDT,AN,dependencies,0',
        '72,-,MR,100,Socher,AN,-,0',
        '73,gigaw,R2,100,APDT,NN,dependencies,0',
        '74,-,R2,100,Socher,NN,-,0',
        '75,gigaw,MR,100,Add,NN,dependencies,0',
        '76,gigaw,MR,100,Mult,NN,dependencies,0',
        '77,gigaw,MR,100,Left,NN,dependencies,0',
        '78,gigaw,MR,100,Right,NN,dependencies,0',
        '79,gigaw,MR,100,Baroni,NN,dependencies,0',
        '80,gigaw,MR,100,Observed,NN,dependencies,0',
        '81,gigaw,MR,100,APDT,NN,dependencies,0',
        '82,-,MR,100,Socher,NN,-,0',
        '83,gigaw,R2,100,Add,AN_NN,dependencies,1',
        '84,gigaw,R2,100,Mult,AN_NN,dependencies,1',
        '85,gigaw,R2,100,Left,AN_NN,dependencies,1',
        '86,gigaw,R2,100,Right,AN_NN,dependencies,1',
        '87,gigaw,R2,100,Baroni,AN_NN,dependencies,1',
        '88,gigaw,R2,100,APDT,AN_NN,dependencies,1',
        '89,-,R2,100,Socher,AN_NN,-,1',
        '90,gigaw,MR,100,Add,AN_NN,dependencies,1',
        '91,gigaw,MR,100,Mult,AN_NN,dependencies,1',
        '92,gigaw,MR,100,Left,AN_NN,dependencies,1',
        '93,gigaw,MR,100,Right,AN_NN,dependencies,1',
        '94,gigaw,MR,100,Baroni,AN_NN,dependencies,1',
        '95,gigaw,MR,100,APDT,AN_NN,dependencies,1',
        '96,-,MR,100,Socher,AN_NN,-,1',
    ]
    for line in table_descr:
        print line
        num, unlab, lab, svd, comp, doc_feats, thes_feats, baronified = line.split(',')
        exp = Experiment(id=num, composer=comp, labelled=lab,
                         unlabelled=unlab, svd=svd,
                         document_features=doc_feats,
                         thesaurus_features=thes_feats,
                         baronified=bool(int(baronified)))
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
            s = StringIO()
            canvas.print_png(s)
            base64_images.append(base64.b64encode(s.getvalue()))
        return base64_images

    @staticmethod
    def get_static_figures(exp_ids):
        return ['static/img/test.jpg']


class ThesisgeneratorExplosionAnalysis(BaseExplosionAnalysis):
    @staticmethod
    def get_tables(exp_ids):
        return [
            ThesisgeneratorExplosionAnalysis.get_performance_table(exp_ids),
            ThesisgeneratorExplosionAnalysis.get_significance_table(exp_ids),
        ] if exp_ids else []

    @staticmethod
    def get_static_figures(exp_ids):
        return [
            "static/figures/stats-exp%d-0.png" % n for n in exp_ids
        ] if exp_ids else []

    @staticmethod
    def get_generated_figures(exp_ids):
        return [
            ThesisgeneratorExplosionAnalysis.get_r2_correlation_plot(exp_ids),
        ] if exp_ids else []

    @staticmethod
    def get_r2_correlation_plot(exp_ids):
        print 'running r2 scatter'
        with open('../thesisgenerator/figures/stats_output.txt') as infile:
            txt = ''.join(infile.readlines())

        # find which experiment a section of the log file relates to
        logs = txt.split('INFO:\t\n\n\n')[1:]
        logs = {int(re.search('DOING EXPERIMENT exp([0-9]+)-0', x).groups()[0]): x for x in logs}

        rsquared = {k: float(re.search('R-squared:\s+([0-9\.]+)', v).groups()[0]) for k, v in logs.items()}
        # todo put this in database

        selected_r2 = [rsquared[n] for n in exp_ids]
        selected_f1 = [get_results_table(n).objects.
                           values_list('score_mean', flat=True).filter(metric='macroavg_f1',
                                                                       classifier='MultinomialNB')[0] for n in exp_ids]

        fig = plt.Figure(dpi=100, facecolor='white')
        ax = fig.add_subplot(111)

        # do whatever magic is needed here
        coef, r2, r2adj = ThesisgeneratorExplosionAnalysis.plot_regression_line(ax,
                                                                                np.array(selected_r2),
                                                                                np.array(selected_f1),
                                                                                np.ones(len(selected_f1)))
        composer_names = []
        for n in exp_ids:
            composer_name = Experiment.objects.values_list('composer', flat=True).filter(id=n)[0]
            composer_names.append('%d-%s' % (n, composer_name))
        ax.scatter(selected_r2, selected_f1)
        for i, txt in enumerate(composer_names):
            ax.annotate(txt, (selected_r2[i], selected_f1[i]), fontsize='x-small')

        if len(coef) > 1:
            fig.suptitle('R2 on class associations vs F1. y=%.2fx%+.2f; r2=%.2f(%.2f)' % (coef[0], coef[1], r2, r2adj))
        else:
            fig.suptitle('All x values are 0, cannot fit regression line')

        canvas = FigureCanvas(fig)
        s = StringIO()
        canvas.print_png(s)
        return base64.b64encode(s.getvalue())

    @staticmethod
    def get_performance_table(exp_ids):
        print 'running performance query'
        data = []
        sample_size = 500
        for n in exp_ids:
            composer_name = Experiment.objects.values_list('composer', flat=True).filter(id=n)[0]
            for classifier in ['MultinomialNB']:
                results = get_results_table(n).objects.all().filter(metric='macroavg_f1',
                                                                    classifier=classifier,
                                                                    sample_size=sample_size)
                if not results.exists():
                    # table or result does not exist
                    print 'skipping table %d and classifier %s' % (n, classifier)
                    continue

                size, f1, f1stderr = results[0].get_performance_info()
                data.append([n, classifier, composer_name,
                             sample_size, '{:.2%}'.format(f1), '{:.2%}'.format(f1stderr)])

        return Table(['id', 'classifier', 'composer', 'sample size', 'score mean', 'std error'],
                     data,
                     'Performance at 500 training documents')

    @staticmethod
    def get_significance_table(exp_ids, classifier='MultinomialNB'):
        # get human-readable labels for the table
        data = []
        composers = []

        print 'Running significance for experiments %r' % exp_ids
        for n in exp_ids:
            # human-readable name
            composer_name = Experiment.objects.values_list('composer', flat=True).filter(id=n)[0]
            cv_folds = get_results_table(n).objects.values_list('cv_folds', flat=True)[0]
            composers.extend(['%d-%s' % (n, composer_name)] * cv_folds)

            # get scores for each CV run- these aren't in the database
            # only at size 500
            outfile = '../thesisgenerator/conf/exp{0}/output/exp{0}-0.out-raw.csv'.format(n)
            df = pd.read_csv(outfile)
            mask = df['classifier'].isin([classifier]) & df['metric'].isin(['macroavg_f1'])
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