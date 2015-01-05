"""does a bunch of plots that I might be interested in and saves images to disk"""

import django_standalone  # do not remove, needed to use the django ORM without a webserver
from gui.user_code import get_demsar_diagram, get_demsar_params
from gui.models import Experiment


def get_interesting_experiments():
    # baseline- signifier
    yield {'decode_handler': 'BaseFeatureHandler'}
    # baseline- hybrid
    yield {'decode_handler': 'SignifierSignifiedFeatureHandler'}

    # random neigh/vect baseline
    yield {'vectors__algorithm__in': ['random_neigh', 'random_vect']}

    # effect of SVD on count vectors, group by feature type
    for feature_type in ['windows', 'dependencies']:
        yield {'labelled': 'amazon_grouped-tagged',
               'vectors__algorithm': 'count_%s' % feature_type,
               'vectors__dimensionality__in': ['0', '100']}

    # count vs windows, with and w/o SVD, group by SVD dims
    for dims in [0, 100]:
        yield {'labelled__in': ['amazon_grouped-tagged'],
               'vectors__algorithm__in': ['count_windows', 'count_dependencies'],
               'vectors__dimensionality__in': dims}

    # using similarity as psedo term count
    yield {'vectors__unlabelled__in': ['gigaw'],
           'neighbour_strategy__in': ['linear'],
           'vectors__rep__in': ['0'],
           'labelled__in': ['reuters21578/r8-tagged-grouped'],
           'document_features__in': ['AN_NN'],
           'vectors__algorithm__in': ['word2vec'],
           'k__in': ['3'],
           'vectors__unlabelled_percentage__in': ['100.0'],
           'use_similarity__in': ['0', '1']}

    # AN vs NN only
    yield {'vectors__algorithm__in': ['word2vec'],
           'vectors__unlabelled_percentage__in': ['100.0'],
           'vectors__rep__in': ['0'],
           'document_features__in': ['AN', 'NN']}

    # some learning curves in an IPython notebook

for i, query_dict in enumerate(get_interesting_experiments()):
    exp_ids = Experiment.objects.values_list('id', flat=True).filter(**query_dict)
    print(exp_ids)
    # get_demsar_diagram(*get_demsar_params(exp_ids),
    # filename='img%d.png' % i)