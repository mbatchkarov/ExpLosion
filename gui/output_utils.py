"""
A set of utilities for manipulating/querying the output of thesisgenerator's classification experiments
"""
import itertools
import numpy as np
from gui.constants import BOOTSTRAP_REPS, CLASSIFIER
from gui.models import Experiment, get_ci


def get_single_vectors_field(exp_id, field_name):
    try:
        vectors = Experiment.objects.get(id=exp_id).expansions.vectors
    except AttributeError:
        vectors = Experiment.objects.get(id=exp_id).clusters.vectors
    return getattr(vectors, field_name) if vectors else None


def get_cv_fold_count(ids):
    return [BOOTSTRAP_REPS] * len(ids)


def get_vectors_field(exp_ids, field_name):
    return np.repeat([get_single_vectors_field(exp_id, field_name) for exp_id in exp_ids],
                     get_cv_fold_count(exp_ids))
    return list(itertools.chain.from_iterable(x))


def get_cv_scores_single_experiment(n, clf=CLASSIFIER):
    _, _, _, bootstrap_scores = get_ci(n, clf=clf)
    return bootstrap_scores


def get_cv_scores_many_experiment(ids, clf=CLASSIFIER):
    scores = [get_cv_scores_single_experiment(i, clf=clf) for i in ids]
    bootstrap_ids = [range(len(x)) for x in scores]
    return list(itertools.chain.from_iterable(scores)), list(itertools.chain.from_iterable(bootstrap_ids))

# data, folds = get_scores([11, 12])
# reps = get_vectors_field([11, 12], 'rep')
# composers = get_vectors_field([11, 12], 'composer')
# percent = get_vectors_field([11, 12], 'unlabelled_percentage')
