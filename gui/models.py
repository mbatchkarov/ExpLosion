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
    # baronified = models.NullBooleanField(null=True)

    def __unicode__(self):
        return 'exp{}:{}-{}-{},{}-{},{}'.format(self.id, self.unlabelled, self.svd, self.thesaurus_features[:3],
                                                self.composer, self.document_features, self.labelled)

    class Meta:
        db_table = 'ExperimentDescriptions'


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