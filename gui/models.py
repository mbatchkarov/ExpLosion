import gzip
import json
from django.db import models
from joblib import Memory
import numpy as np
from sklearn.metrics import accuracy_score
from gui.constants import BOOTSTRAP_REPS, CLASSIFIER, METRIC_DB

memory = Memory(cachedir='.', verbose=0)


class Expansions(models.Model):
    vectors = models.OneToOneField('Vectors', null=True, default=None, on_delete='SET NULL', related_name='vectors')
    entries_of = models.OneToOneField('Vectors', null=True, default=None, on_delete='SET NULL',
                                      related_name='entries_of')
    allow_overlap = models.BooleanField(default=False)  # allow lexical overlap between features and its replacements
    use_random_neighbours = models.BooleanField(default=False)
    decode_handler = models.CharField(max_length=255,
                                      default='SignifiedOnlyFeatureHandler')  # signifier, signified, hybrid
    k = models.IntegerField(default=3)  # how many neighbours entries are replaced with at decode time
    noise = models.FloatField(default=0)
    use_similarity = models.BooleanField(default=False)  # use phrase sim as pseudo term count
    neighbour_strategy = models.CharField(max_length=255,
                                          default='linear')  # how neighbours are found- linear or skipping strategy

    class Meta:
        managed = False
        db_table = 'expansions'


class Clusters(models.Model):
    num_clusters = models.IntegerField(null=True, default=None)
    path = models.CharField(max_length=255, null=True, default=None)
    # vectors must be consistent with Experiment.vectors
    vectors = models.OneToOneField('Vectors', null=True, default=None, on_delete='SET NULL')

    class Meta:
        managed = False
        db_table = 'clusters'

    def __str__(self):
        return 'CL:%d-vec%d' % (self.num_clusters, self.vectors.id)


class Experiment(models.Model):
    id = models.IntegerField(primary_key=True)
    document_features_tr = models.CharField(max_length=255, default='J+N+AN+NN')  # AN+NN, AN only, NN only, ...
    document_features_ev = models.CharField(max_length=255, default='AN+NN')
    labelled = models.CharField(max_length=255, )  # name/path of labelled corpus used
    clusters = models.OneToOneField(Clusters, null=True, default=None, on_delete='SET NULL',
                                    related_name='clusters')
    expansions = models.OneToOneField(Expansions, null=True, default=None, on_delete='SET NULL',
                                      related_name='expansions')

    date_ran = models.DateField(null=True, default=None)
    git_hash = models.CharField(max_length=255, null=True, default=None)
    minutes_taken = models.FloatField(null=True, default=None)

    class Meta:
        managed = False
        db_table = 'classificationexperiment'


class _Dummy():
    def ci(self, *args, **kwargs):
        return 0, 0, 0, [0] * BOOTSTRAP_REPS


class GetOrZeroManager(models.Manager):
    """
    Adds get_or_none method to objects
    Source http://stackoverflow.com/a/2021833/419338
    """

    def get_or_zero(self, **kwargs):
        try:
            return self.get(**kwargs)
        except self.model.DoesNotExist:
            print('No results for query', kwargs)
            return _Dummy()


class Results(models.Model):
    id = models.OneToOneField(Experiment, primary_key=True)
    classifier = models.CharField(max_length=255)
    accuracy_mean = models.FloatField()
    accuracy_std = models.FloatField()
    microf1_mean = models.FloatField()
    microf1_std = models.FloatField()
    macrof1_mean = models.FloatField()
    macrof1_std = models.FloatField()
    _predictions = models.BinaryField(null=True)
    _gold = models.BinaryField(null=True)

    objects = GetOrZeroManager()

    def get_performance_info(self, metric=METRIC_DB):
        return round(getattr(self, '%s_mean' % metric), 6), \
               round(getattr(self, '%s_std' % metric), 6)

    class Meta:
        managed = False
        db_table = 'results'

    @property
    def predictions(self):
        return np.array(json.loads(gzip.decompress(self._predictions).decode('utf8')))

    @property
    def gold(self):
        return np.array(json.loads(gzip.decompress(self._gold).decode('utf8')))

    def ci(self, nboot=BOOTSTRAP_REPS, statistic=accuracy_score):
        print('Calculating CI for exp', self.id.id, flush=True)
        gold = self.gold
        predictions = self.predictions

        scores = []
        for _ in range(nboot):
            ind = np.random.choice(range(len(gold)), size=len(gold))
            g = gold[ind]
            pred = predictions[ind]
            f1 = statistic(g, pred)
            scores.append(f1)
        scores = np.array(sorted(scores))
        self.bootstrap_scores = scores
        self.mean = scores.mean()
        self.low = np.percentile(scores, 2.5)
        self.high = np.percentile(scores, 97.5)
        return self.mean, self.low, self.high, self.bootstrap_scores


@memory.cache
def get_ci(exp_id, clf=CLASSIFIER):
    # django creates a new object for each query, which causes the expensive
    # ci() method to be called multiple times. Cache that call to save time
    # NB: can't use cache on another object, that's why we use this function
    # also, return 0 if no results for this experiment. that way plotting code will
    # at least run, but the error will be visible
    return Results.objects.get_or_zero(id=exp_id, classifier=clf).ci()


class FullResults(models.Model):
    id = models.OneToOneField(Experiment, primary_key=True)
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
    unlabelled_percentage = models.IntegerField(blank=True, null=True)
    unlabelled = models.CharField(max_length=255, blank=True)
    path = models.CharField(max_length=255, blank=True)
    composer = models.CharField(max_length=255)
    modified = models.DateField(blank=True, null=True)
    size = models.IntegerField(blank=True, null=True)
    rep = models.IntegerField()

    def __str__(self):
        fields = ','.join((str(x) for x in [self.algorithm, self.composer,
                                            self.dimensionality,
                                            self.unlabelled_percentage]))
        return 'Vectors %d:' % self.id + fields

    class Meta:
        managed = False
        db_table = 'vectors'
