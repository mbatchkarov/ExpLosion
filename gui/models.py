import gzip
import json
from django.db import models
from joblib import Memory
import numpy as np
from sklearn.metrics import accuracy_score

memory = Memory(cachedir='.', verbose=0)


class Experiment(models.Model):
    id = models.IntegerField(primary_key=True)
    document_features_tr = models.CharField(max_length=255)  # AN+NN, AN only, NN only, ...
    document_features_ev = models.CharField(max_length=255)
    use_similarity = models.IntegerField()
    use_random_neighbours = models.IntegerField()
    decode_handler = models.CharField(max_length=255)
    vectors = models.OneToOneField('Vectors', blank=True, null=True)
    labelled = models.CharField(max_length=255)
    date_ran = models.DateField(blank=True, null=True)
    git_hash = models.CharField(max_length=255, blank=True)
    k = models.IntegerField()  # how many neighbours entries are replaced with at decode time
    neighbour_strategy = models.CharField(max_length=255)
    noise = models.FloatField(default=0)

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
            self.neighbour_strategy == other.neighbour_strategy and
            abs(self.noise - other.noise) < 0.001
        )

    def __hash__(self):
        return hash((self.document_features, self.use_similarity,
                     self.use_random_neighbours, self.decode_handler,
                     self.vectors.id if self.vectors else None, self.labelled,
                     self.k, self.neighbour_strategy, self.noise))

    class Meta:
        managed = False
        db_table = 'classificationexperiment'


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


    def get_performance_info(self, metric='macrof1'):
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

    def ci(self, nboot=500, statistic=accuracy_score):
        print('Calculating CI for exp', self.id)
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
def get_ci(exp_id, clf='MultinomialNB'):
    # django creates a new object for each query, which causes the expensive
    # ci() method to be called multiple times. Cache that call to save time
    # NB: can't use cache on another object, that's why we use this function
    return Results.objects.get(id=exp_id, classifier=clf).ci()


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
    unlabelled_percentage = models.FloatField(blank=True, null=True)
    unlabelled = models.CharField(max_length=255, blank=True)
    path = models.CharField(max_length=255, blank=True)
    composer = models.CharField(max_length=255)
    modified = models.DateField(blank=True, null=True)
    size = models.IntegerField(blank=True, null=True)
    rep = models.IntegerField()
    use_ppmi = models.BooleanField(default=0)

    def __str__(self):
        fields = ','.join((str(x) for x in [self.algorithm, self.composer,
                                            self.dimensionality,
                                            self.unlabelled_percentage]))
        return 'Vectors %d:' % self.id + fields

    class Meta:
        managed = False
        db_table = 'vectors'
