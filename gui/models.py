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
    k = models.IntegerField()  # how many neighbours entries are replaced with at decode time
    neighbour_strategy = models.CharField(max_length=255)

    def __str__(self):
        basic_settings = ','.join((str(x) for x in [self.labelled, self.vectors]))
        return '%s: %s' % (self.id, basic_settings)

    def __eq__(self, other):
        if not isinstance(other, Experiment):
            return False
        return (
            self.document_features == other.document_features and
            self.use_similarity == other.use_similarity and
            self.use_random_neighbours == other.use_random_neighbours and
            self.decode_handler == other.decode_handler and
            self.vectors.id == other.vectors.id and
            self.labelled == other.labelled and
            self.k == other.k and
            self.neighbour_strategy == other.neighbour_strategy
        )

    def __hash__(self):
        return hash((self.document_features, self.use_similarity,
                     self.use_random_neighbours, self.decode_handler,
                     self.vectors.id if self.vectors else None, self.labelled,
                     self.k, self.neighbour_strategy))

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

    def get_performance_info(self, metric='macrof1'):
        return round(getattr(self, '%s_mean' % metric), 6), \
               round(getattr(self, '%s_std' % metric), 6)

    class Meta:
        managed = False
        db_table = 'results'


class FullResults(models.Model):
    id = models.ForeignKey(Experiment, primary_key=True)
    classifier = models.CharField(max_length=255)
    cv_fold = models.IntegerField()
    accuracy_score = models.FloatField()
    macroavg_f1 = models.FloatField()
    microavg_f1 = models.FloatField()
    macroavg_rec = models.FloatField()
    microavg_rec = models.FloatField()
    microavg_prec = models.FloatField()
    macroavg_prec = models.FloatField()

    class Meta:
        managed = False
        db_table = 'fullresults'


class Vectors(models.Model):
    id = models.IntegerField(primary_key=True)
    algorithm = models.CharField(max_length=255)
    dimensionality = models.IntegerField(blank=True, null=True)
    unlabelled_percentage = models.FloatField(blank=True, null=True)
    unlabelled = models.CharField(max_length=255, blank=True)
    path = models.CharField(max_length=255, blank=True)
    composer = models.CharField(max_length=255)
    modified = models.DateField(blank=True, null=True)
    size = models.IntegerField(blank=True, null=True)
    rep = models.IntegerField()

    def __str__(self):
        return 'Vectors: ' + ','.join(str(x) for x in [self.algorithm, self.composer, self.dimensionality])

    class Meta:
        managed = False
        db_table = 'vectors'


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
