"""
A set of utilities for manipulating/querying the output of thesisgenerator's classification experiments
"""
import itertools
import numpy as np
from gui.models import Experiment, Results, get_ci

METRIC_DB = 'macrof1'
METRIC_CSV_FILE = 'macroavg_f1'
BOOTSTRAP_REPS = 500


def get_single_vectors_field(exp_id, field_name):
    vectors = Experiment.objects.get(id=exp_id).vectors
    return getattr(vectors, field_name) if vectors else None


def get_cv_fold_count(ids):
    res = []
    for i in ids:
        # must use the second parameter, otherwise there will be a cache miss
        # apparently get_ci(1), get_ci(1, 1) and get_ci(1, clf=1) are different calls
        _, _, _, bootstrap_scores = get_ci(i, clf='MultinomialNB')
        res.append(len(bootstrap_scores))
    return res


def get_vectors_field(exp_ids, field_name):
    return np.repeat([get_single_vectors_field(exp_id, field_name) for exp_id in exp_ids],
                     get_cv_fold_count(exp_ids))
    return list(itertools.chain.from_iterable(x))


def get_cv_scores_single_experiment(n, clf='MultinomialNB'):
    _, _, _, bootstrap_scores = get_ci(n, clf=clf)
    return bootstrap_scores


def get_cv_scores_many_experiment(ids, clf='MultinomialNB'):
    scores = [get_cv_scores_single_experiment(i, clf=clf) for i in ids]
    bootstrap_ids = [range(len(x)) for x in scores]
    return list(itertools.chain.from_iterable(scores)), list(itertools.chain.from_iterable(bootstrap_ids))

# data, folds = get_scores([11, 12])
# reps = get_vectors_field([11, 12], 'rep')
# composers = get_vectors_field([11, 12], 'composer')
# percent = get_vectors_field([11, 12], 'unlabelled_percentage')

