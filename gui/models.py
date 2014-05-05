from math import sqrt
from operator import itemgetter
from django.db import models


class Experiment(models.Model):
    id = models.PositiveIntegerField(blank=False, null=False, primary_key=True)
    composer = models.CharField(max_length=100)
    labelled = models.CharField(max_length=100)
    unlabelled = models.CharField(max_length=100)
    thesaurus_features = models.CharField(max_length=100)
    svd = models.IntegerField()
    document_features = models.CharField(max_length=100)
    baronified = models.IntegerField()

    def __unicode__(self):
        return 'exp{}:{}-{}-{},{}-{},{}'.format(self.id, self.unlabelled, self.svd, self.thesaurus_features[:3],
                                                self.composer, self.document_features, self.labelled)

    class Meta:
        db_table = 'ExperimentDescriptions'


def get_results_table(param):
    """
    Thesisgenerator-specific model for extracting results out of the database
    From http://stackoverflow.com/questions/5036357/single-django-model-multiple-tables
    """

    class MyClassMetaclass(models.base.ModelBase):
        def __new__(cls, name, bases, attrs):
            return models.base.ModelBase.__new__(cls, 'data%d' % param, bases, attrs)

    class ThesisgeneratorPerformanceResult(models.Model):
        __metaclass__ = MyClassMetaclass

        id = models.AutoField(primary_key=True)
        name = models.TextField(blank=True)
        git_hash = models.TextField(blank=True)
        consolidation_date = models.DateTimeField()
        cv_folds = models.IntegerField(blank=True, null=True)
        sample_size = models.IntegerField(blank=True, null=True)
        classifier = models.TextField(blank=True)
        metric = models.TextField(blank=True)
        score_mean = models.FloatField(blank=True, null=True)
        score_std = models.FloatField(blank=True, null=True)

        class Meta:
            db_table = 'data%d' % param
            ordering = ['sample_size']

        def get_performance_info(self):
            return self.sample_size, self.score_mean, self.score_std / sqrt(self.cv_folds)

    return ThesisgeneratorPerformanceResult


class Table():
    def __init__(self, header, rows, desc):
        for row in rows:
            if not len(header) == len(row):
                raise ValueError('Malformed table. Header has %d columns and one of '
                                 'the rows has %d' % (len(header), len(row)))

        self.header = header
        self.rows = rows
        self.description = desc

    def prune(self):
        """
        Removes columns where all values are duplicates
        """
        dupl_idx = []
        for i, column_name in enumerate(self.header):
            if len(set(row[i] for row in self.rows)) == 1:
                # just one value
                dupl_idx.append(i)
        if dupl_idx and len(self.rows) > 1:
            idx_to_keep = set(range(len(self.header))) - set(dupl_idx)
            self.header = itemgetter(*idx_to_keep)(self.header)
            self.rows = [itemgetter(*idx_to_keep)(row) for row in self.rows]