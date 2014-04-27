from gui.models import Experiment
import matplotlib.pyplot as plt
from cStringIO import StringIO
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


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
        '61,Neuro,R2,0,Socher,AN_NN,Neuro,0',
        '62,Neuro,MR,0,Socher,AN_NN,Neuro,0',
        '63,gigaw,R2,100,APDT,AN,dependencies,0',
        '64,gigaw,R2,100,Socher,AN,dependencies,0',
        '65,gigaw,MR,100,Add,AN,dependencies,0',
        '66,gigaw,MR,100,Mult,AN,dependencies,0',
        '67,gigaw,MR,100,Left,AN,dependencies,0',
        '68,gigaw,MR,100,Right,AN,dependencies,0',
        '69,gigaw,MR,100,Baroni,AN,dependencies,0',
        '70,gigaw,MR,100,Observed,AN,dependencies,0',
        '71,gigaw,MR,100,APDT,AN,dependencies,0',
        '72,gigaw,MR,100,Socher,AN,dependencies,0',
        '73,gigaw,R2,100,APDT,NN,dependencies,0',
        '74,gigaw,R2,100,Socher,NN,dependencies,0',
        '75,gigaw,MR,100,Add,NN,dependencies,0',
        '76,gigaw,MR,100,Mult,NN,dependencies,0',
        '77,gigaw,MR,100,Left,NN,dependencies,0',
        '78,gigaw,MR,100,Right,NN,dependencies,0',
        '79,gigaw,MR,100,Baroni,NN,dependencies,0',
        '80,gigaw,MR,100,Observed,NN,dependencies,0',
        '81,gigaw,MR,100,APDT,NN,dependencies,0',
        '82,gigaw,MR,100,Socher,NN,dependencies,0',
        '83,gigaw,R2,100,Add,AN_NN,dependencies,1',
        '84,gigaw,R2,100,Mult,AN_NN,dependencies,1',
        '85,gigaw,R2,100,Left,AN_NN,dependencies,1',
        '86,gigaw,R2,100,Right,AN_NN,dependencies,1',
        '87,gigaw,R2,100,Baroni,AN_NN,dependencies,1',
        '88,gigaw,R2,100,APDT,AN_NN,dependencies,1',
        '89,gigaw,R2,100,Socher,AN_NN,dependencies,1',
        '90,gigaw,MR,100,Add,AN_NN,dependencies,1',
        '91,gigaw,MR,100,Mult,AN_NN,dependencies,1',
        '92,gigaw,MR,100,Left,AN_NN,dependencies,1',
        '93,gigaw,MR,100,Right,AN_NN,dependencies,1',
        '94,gigaw,MR,100,Baroni,AN_NN,dependencies,1',
        '95,gigaw,MR,100,APDT,AN_NN,dependencies,1',
        '96,gigaw,MR,100,Socher,AN_NN,dependencies,1',
        '97,gigaw,R2,100,Random,AN_NN,dependencies,0',
        '98,gigaw,MR,100,Random,AN_NN,dependencies,0',
    ]
    for line in table_descr:
        print line
        num, unlab, lab, svd, comp, doc_feats, thes_feats, baronified = line.split(',')
        exp = Experiment(number=num, composer=comp, labelled=lab,
                         unlabelled=unlab, svd=svd,
                         document_features=doc_feats,
                         thesaurus_features=thes_feats,
                         baronified=bool(int(baronified)))
        exp.save()


class Table():
    def __init__(self, header, rows, desc):
        self.header = header
        self.rows = rows
        self.description = desc


class BaseExplosionAnalysis(object):
    def __init__(self):
        self.analyses = []

    @staticmethod
    def populate_experiments_db(*args, **kwargs):
        if not Experiment.objects.count():
            Experiment.objects.all().delete()
            populate_manually()

    @staticmethod
    def get_tables():
        return [
            Table(['num', 'val', 'val', 'val'],
                  [[1, 'test', 1, 1], [2, 'test', 2, 2]],
                  'some table'),
            Table(['num', 'val', 'val', 'val'],
                  [[1, 'test', 1, 1], [2, 'test', 2, 2]],
                  'some other table')
        ]

    @staticmethod
    def get_figures():
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
