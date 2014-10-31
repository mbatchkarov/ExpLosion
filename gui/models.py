from math import sqrt
from operator import itemgetter
from django.db import models


class Experiment(models.Model):
    id = models.IntegerField(primary_key=True)
    document_features = models.CharField(max_length=255)
    use_similarity = models.IntegerField()
    use_random_neighbours = models.IntegerField()
    decode_handler = models.CharField(max_length=255)
    vectors = models.ForeignKey('Vectors', blank=True, null=True)
    labelled = models.CharField(max_length=255)
    date_ran = models.DateField(blank=True, null=True)
    git_hash = models.CharField(max_length=255, blank=True)

    def __str__(self):
        basic_settings = ','.join((str(x) for x in [self.labelled, self.vectors]))
        return '%s: %s' % (self.id, basic_settings)

    class Meta:
        managed = False
        db_table = 'classificationexperiment'


class Results(models.Model):
    id = models.ForeignKey(Experiment, primary_key=True)
    classifier = models.CharField(max_length=255)
    accuracy_mean = models.FloatField()
    accuracy_std = models.FloatField()
    microf1_mean = models.FloatField()
    microf1_std = models.FloatField()
    macrof1_mean = models.FloatField()
    macrof1_std = models.FloatField()

    def get_performance_info(self, metric):
        return getattr(self, '%s_mean' % metric), getattr(self, '%s_std' % metric)

    class Meta:
        managed = False
        db_table = 'results'


class Vectors(models.Model):
    id = models.IntegerField(primary_key=True)
    algorithm = models.CharField(max_length=255)
    dimensionality = models.IntegerField(blank=True, null=True)
    unlabelled_percentage = models.IntegerField(blank=True, null=True)
    unlabelled = models.CharField(max_length=255, blank=True)
    path = models.CharField(max_length=255, blank=True)
    composer = models.CharField(max_length=255)
    modified = models.DateField(blank=True, null=True)
    size = models.IntegerField(blank=True, null=True)

    def __str__(self):
        return 'Vectors: ' + ','.join(str(x) for x in [self.algorithm, self.composer, self.dimensionality])

    class Meta:
        managed = False
        db_table = 'vectors'


cached_models = {}


# def get_results_table(param):
# """
#     Thesisgenerator-specific model for extracting results out of the database
#     From http://stackoverflow.com/questions/5036357/single-django-model-multiple-tables
#     """
#
#     class MyClassMetaclass(models.base.ModelBase):
#         def __new__(cls, name, bases, attrs):
#             if param not in cached_models:
#                 cached_models[param] = models.base.ModelBase.__new__(cls, 'data%d' % param, bases, attrs)
#             return cached_models[param]
#
#     class ThesisgeneratorPerformanceResult(models.Model, metaclass=MyClassMetaclass):
#         __metaclass__ = MyClassMetaclass
#
#         id = models.AutoField(primary_key=True)
#         name = models.TextField(blank=True)
#         git_hash = models.TextField(blank=True)
#         consolidation_date = models.DateTimeField()
#         cv_folds = models.IntegerField(blank=True, null=True)
#         sample_size = models.IntegerField(blank=True, null=True)
#         classifier = models.TextField(blank=True)
#         metric = models.TextField(blank=True)
#         score_mean = models.FloatField(blank=True, null=True)
#         score_std = models.FloatField(blank=True, null=True)
#
#         class Meta:
#             db_table = 'data%d' % param
#             ordering = ['sample_size']
#
#
#
#     return ThesisgeneratorPerformanceResult


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
            if len(idx_to_keep) == 1:
                # itemgetter will return just an item, no a list containing the item
                self.header = [self.header]
                self.rows = [[x] for x in self.rows]
