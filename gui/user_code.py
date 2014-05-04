from gui.models import Experiment
import matplotlib.pyplot as plt
from cStringIO import StringIO
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def populate_manually():
    # run manually in django console to populate the database
    table_descr = [
        '1,gigaw,R2,0,Add,AN_NN,dependencies',
        '2,gigaw,R2,0,Mult,AN_NN,dependencies',
        '3,gigaw,R2,0,Left,AN_NN,dependencies',
        '4,gigaw,R2,0,Right,AN_NN,dependencies',
        '5,gigaw,R2,0,Observed,AN_NN,dependencies',
        '6,gigaw,R2,100,Add,AN_NN,dependencies',
        '7,gigaw,R2,100,Mult,AN_NN,dependencies',
        '8,gigaw,R2,100,Left,AN_NN,dependencies',
        '9,gigaw,R2,100,Right,AN_NN,dependencies',
        '10,gigaw,R2,100,Baroni,AN_NN,dependencies',
        '11,gigaw,R2,100,Observed,AN_NN,dependencies',
        '12,gigaw,MR,0,Add,AN_NN,dependencies',
        '13,gigaw,MR,0,Mult,AN_NN,dependencies',
        '14,gigaw,MR,0,Left,AN_NN,dependencies',
        '15,gigaw,MR,0,Right,AN_NN,dependencies',
        '16,gigaw,MR,0,Observed,AN_NN,dependencies',
        '17,gigaw,MR,100,Add,AN_NN,dependencies',
        '18,gigaw,MR,100,Mult,AN_NN,dependencies',
        '19,gigaw,MR,100,Left,AN_NN,dependencies',
        '20,gigaw,MR,100,Right,AN_NN,dependencies',
        '21,gigaw,MR,100,Baroni,AN_NN,dependencies',
        '22,gigaw,MR,100,Observed,AN_NN,dependencies',
        '23,gigaw,R2,0,Add,AN_NN,windows',
        '24,gigaw,R2,0,Mult,AN_NN,windows',
        '25,gigaw,R2,0,Left,AN_NN,windows',
        '26,gigaw,R2,0,Right,AN_NN,windows',
        '27,gigaw,R2,0,Observed,AN_NN,windows',
        '28,gigaw,R2,100,Add,AN_NN,windows',
        '29,gigaw,R2,100,Mult,AN_NN,windows',
        '30,gigaw,R2,100,Left,AN_NN,windows',
        '31,gigaw,R2,100,Right,AN_NN,windows',
        '32,gigaw,R2,100,Baroni,AN_NN,windows',
        '33,gigaw,R2,100,Observed,AN_NN,windows',
        '34,gigaw,MR,0,Add,AN_NN,windows',
        '35,gigaw,MR,0,Mult,AN_NN,windows',
        '36,gigaw,MR,0,Left,AN_NN,windows',
        '37,gigaw,MR,0,Right,AN_NN,windows',
        '38,gigaw,MR,0,Observed,AN_NN,windows',
        '39,gigaw,MR,100,Add,AN_NN,windows',
        '40,gigaw,MR,100,Mult,AN_NN,windows',
        '41,gigaw,MR,100,Left,AN_NN,windows',
        '42,gigaw,MR,100,Right,AN_NN,windows',
        '43,gigaw,MR,100,Baroni,AN_NN,windows',
        '44,gigaw,MR,100,Observed,AN_NN,windows',
        '45,gigaw,R2,100,Add,AN,dependencies',
        '46,gigaw,R2,100,Mult,AN,dependencies',
        '47,gigaw,R2,100,Left,AN,dependencies',
        '48,gigaw,R2,100,Right,AN,dependencies',
        '49,gigaw,R2,100,Baroni,AN,dependencies',
        '50,gigaw,R2,100,Observed,AN,dependencies',
        '51,gigaw,R2,100,Add,NN,dependencies',
        '52,gigaw,R2,100,Mult,NN,dependencies',
        '53,gigaw,R2,100,Left,NN,dependencies',
        '54,gigaw,R2,100,Right,NN,dependencies',
        '55,gigaw,R2,100,Baroni,NN,dependencies',
        '56,gigaw,R2,100,Observed,NN,dependencies',
        '57,gigaw,R2,0,APDT,AN_NN,dependencies',
        '58,gigaw,R2,100,APDT,AN_NN,dependencies',
        '59,gigaw,MR,0,APDT,AN_NN,dependencies',
        '60,gigaw,MR,100,APDT,AN_NN,dependencies',
        '61,-,R2,100,Socher,AN_NN,-',
        '62,-,MR,100,Socher,AN_NN,-',
        '63,gigaw,R2,100,APDT,AN,dependencies',
        '64,-,R2,100,Socher,AN,-',
        '65,gigaw,MR,100,Add,AN,dependencies',
        '66,gigaw,MR,100,Mult,AN,dependencies',
        '67,gigaw,MR,100,Left,AN,dependencies',
        '68,gigaw,MR,100,Right,AN,dependencies',
        '69,gigaw,MR,100,Baroni,AN,dependencies',
        '70,gigaw,MR,100,Observed,AN,dependencies',
        '71,gigaw,MR,100,APDT,AN,dependencies',
        '72,-,MR,100,Socher,AN,-',
        '73,gigaw,R2,100,APDT,NN,dependencies',
        '74,-,R2,100,Socher,NN,-',
        '75,gigaw,MR,100,Add,NN,dependencies',
        '76,gigaw,MR,100,Mult,NN,dependencies',
        '77,gigaw,MR,100,Left,NN,dependencies',
        '78,gigaw,MR,100,Right,NN,dependencies',
        '79,gigaw,MR,100,Baroni,NN,dependencies',
        '80,gigaw,MR,100,Observed,NN,dependencies',
        '81,gigaw,MR,100,APDT,NN,dependencies',
        '82,-,MR,100,Socher,NN,-',
        '83,gigaw,R2,100,Add,AN_NN,dependencies',
        '84,gigaw,R2,100,Mult,AN_NN,dependencies',
        '85,gigaw,R2,100,Left,AN_NN,dependencies',
        '86,gigaw,R2,100,Right,AN_NN,dependencies',
        '87,gigaw,R2,100,Baroni,AN_NN,dependencies',
        '88,gigaw,R2,100,APDT,AN_NN,dependencies',
        '89,-,R2,100,Socher,AN_NN,-',
        '90,gigaw,MR,100,Add,AN_NN,dependencies',
        '91,gigaw,MR,100,Mult,AN_NN,dependencies',
        '92,gigaw,MR,100,Left,AN_NN,dependencies',
        '93,gigaw,MR,100,Right,AN_NN,dependencies',
        '94,gigaw,MR,100,Baroni,AN_NN,dependencies',
        '95,gigaw,MR,100,APDT,AN_NN,dependencies',
        '96,-,MR,100,Socher,AN_NN,-',
    ]
    for line in table_descr:
        print line
        num, unlab, lab, svd, comp, doc_feats, thes_feats = line.split(',') # todo add baronified
        exp = Experiment(id=num, composer=comp, labelled=lab,
                         unlabelled=unlab, svd=svd,
                         document_features=doc_feats,
                         thesaurus_features=thes_feats,)
                         # baronified=bool(int(baronified)))
        exp.save()


class Table():
    def __init__(self, header, rows, desc):
        for row in rows:
            if not len(header) == len(row):
                raise ValueError('Malformed table. Header has %d columns and one of '
                                 'the rows has %d'%(len(header), len(row)))

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
