from django.db import models


class Experiment(models.Model):
    number = models.PositiveIntegerField(blank=False, null=False, primary_key=True)
    composer = models.CharField(max_length=100)
    labelled = models.CharField(max_length=100)
    unlabelled = models.CharField(max_length=100)
    thesaurus_features = models.CharField(max_length=100)
    svd = models.IntegerField()
    document_features = models.CharField(max_length=100)
    # baronified = models.NullBooleanField(null=True)

    def __unicode__(self):
        return 'exp{}:{}-{}-{},{}-{},{}'.format(self.number, self.unlabelled, self.svd, self.thesaurus_features[:3],
                                                self.composer, self.document_features, self.labelled)

    class Meta:
        db_table = 'ExperimentDescriptions'
