"""does a bunch of plots that I might be interested in and saves images to disk"""

import django_standalone  # do not remove, needed to use the django ORM without a webserver
from gui.user_code import get_demsar_diagram, get_demsar_params
from gui.models import Experiment


def get_interesting_experiments():
    # baseline- signifier. That didn't work for TechTC data, so ignore it
    yield {'decode_handler': 'BaseFeatureHandler',
           'labelled__in': ['amazon_grouped-tagged', 'reuters21578/r8-tagged-grouped']}, ['labelled']
    # baseline- hybrid
    yield {'decode_handler': 'SignifierSignifiedFeatureHandler'}, None

    # random neigh/vect baseline
    yield {'vectors__algorithm__in': ['random_neigh', 'random_vect']}, ['labelled', 'vectors__algorithm']

    # effect of SVD on count vectors, group by feature type
    for feature_type in ['windows', 'dependencies']:
        yield {'labelled': 'amazon_grouped-tagged',
               'vectors__algorithm': 'count_%s' % feature_type,
               'vectors__dimensionality__in': ['0', '100']}, \
              ['vectors__algorithm', 'vectors__dimensionality', 'vectors__composer']

    # count vs windows, with and w/o SVD, group by SVD dims
    for dims in [0, 100]:
        yield {'labelled': 'amazon_grouped-tagged',
               'vectors__algorithm__in': ['count_windows', 'count_dependencies'],
               'vectors__dimensionality': dims}, \
              ['vectors__algorithm', 'vectors__dimensionality', 'vectors__composer']

    # NOTE: THESE HAVE ONLY BEEN DONE ON REUTERS DATA, probably unreliable?
    # using similarity as psedo term count
    yield {'vectors__unlabelled': 'gigaw',
           'neighbour_strategy': 'linear',
           'vectors__rep': '0',
           'labelled': 'reuters21578/r8-tagged-grouped',
           'document_features': 'AN_NN',
           'vectors__algorithm': 'word2vec',
           'k': '3',
           'vectors__unlabelled_percentage': '100.0',
           'use_similarity__in': ['0', '1']}, None

    # AN vs NN only
    yield {'vectors__algorithm': 'word2vec',
           'vectors__unlabelled_percentage': '100.0',
           'vectors__rep': '0',
           'document_features__in': ['AN', 'NN']}, None

    # some learning curves in an IPython notebook


for i, (query_dict, fields) in enumerate(get_interesting_experiments()):
    default_fields = ['vectors__algorithm', 'vectors__composer']
    exp_ids = Experiment.objects.values_list('id', flat=True).filter(**query_dict)
    print(exp_ids)
    get_demsar_diagram(*get_demsar_params(exp_ids, name_format=fields if fields else default_fields),
                       filename='img%d.png' % i)