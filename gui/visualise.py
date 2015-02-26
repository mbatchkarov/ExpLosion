"""does a bunch of plots that I might be interested in and saves images to disk"""

import django_standalone  # do not remove, needed to use the django ORM without a webserver
from gui.user_code import get_demsar_diagram, get_demsar_params
from gui.models import Experiment


def get_interesting_experiments():
    # baseline- signifier. That didn't work for TechTC data, so ignore it
    yield {'decode_handler': 'BaseFeatureHandler',
           'labelled__in': ['amazon_grouped-tagged', 'aclImdb',
                            'reuters21578/r8-tagged-grouped']}, ['labelled']
    print('------------------------------------')
    # baseline- hybrid
    yield {'decode_handler': 'SignifierSignifiedFeatureHandler',
           'labelled': 'amazon_grouped-tagged'}, None
    print('------------------------------------')

    # random neigh/vect baseline
    yield {'vectors__algorithm__in': ['random_neigh', 'random_vect']}, ['labelled', 'vectors__algorithm']
    print('------------------------------------')

    # effect of SVD on count vectors, group by feature type
    for feature_type in ['windows', 'dependencies']:
        yield {'labelled': 'amazon_grouped-tagged',
               'vectors__algorithm': 'count_%s' % feature_type,
               'vectors__dimensionality__in': [0, 100]}, \
              ['vectors__algorithm', 'vectors__dimensionality', 'vectors__composer']
    print('------------------------------------')

    # count vs windows, with and w/o SVD, group by SVD dims
    for dims in [0, 100]:
        yield {'labelled': 'amazon_grouped-tagged',
               'vectors__algorithm__in': ['count_windows', 'count_dependencies'],
               'vectors__dimensionality': dims}, \
              ['vectors__algorithm', 'vectors__dimensionality', 'vectors__composer']
    print('------------------------------------')

    # NOTE: THESE HAVE ONLY BEEN DONE ON REUTERS DATA, probably unreliable?
    # using similarity as psedo term count
    yield {'vectors__unlabelled': 'gigaw',
           'neighbour_strategy': 'linear',
           'vectors__rep': 0,
           'labelled': 'reuters21578/r8-tagged-grouped',
           'document_features': 'AN_NN',
           'vectors__algorithm': 'word2vec',
           'k': 3,
           'vectors__unlabelled_percentage': 100.0,
           'use_similarity__in': [0, 1]}, ['labelled', 'vectors__composer', 'use_similarity']
    print('------------------------------------')
    print('fuck')
    # AN vs NN only
    for composers in [['Add', 'Mult'], ['Left', 'Right']]:
        yield {'vectors__algorithm': 'word2vec',
               'vectors__composer__in': composers,
               'labelled': 'reuters21578/r8-tagged-grouped',
               'vectors__unlabelled_percentage': '100.0',
               'vectors__rep': 0,
               'neighbour_strategy': 'linear',
               'use_similarity': 0, 'k': 3,
               'decode_handler': 'SignifiedOnlyFeatureHandler',
               'document_features__in': ['AN', 'NN', 'AN_NN']}, ['labelled', 'vectors__composer', 'document_features']
    print('------------------------------------')

    # best count models (deps, no SVD, PPMI) vs w2v
    yield {'vectors__composer__in': ['Add', 'Left', 'Socher'],
           'vectors__algorithm__in': ['count_dependencies', 'word2vec'],
           'labelled': 'amazon_grouped-tagged',
           'document_features': 'AN_NN',
           'neighbour_strategy': 'linear',
           'k': 3,
           'use_similarity': 0,
           'vectors__unlabelled_percentage': 100.0,
           'decode_handler': 'SignifiedOnlyFeatureHandler',
           'vectors__dimensionality': 100,
           # todo this is wrong, we want 0 for dependency vectors. Do this right by hand
           'vectors__rep': 0}, None
    print('------------------------------------')

    # best models (according to Amazon task) at MR task ...
    yield {'vectors__algorithm__in': ['word2vec', 'count_dependencies'],
           'labelled': 'movie-reviews-tagged',
           'vectors__dimensionality': 100}, None

    # ... and at one of the TechTC tasks
    yield {'vectors__algorithm__in': ['word2vec', 'count_dependencies'],
           'labelled': 'techtc100-clean/Exp_186330_94142-tagged',
           'vectors__dimensionality': 100}, None

    # Demsar diagram for Turian vectors with add-mult-left-righ-socher composers
    yield {'vectors__unlabelled': 'turian',
           'labelled': 'amazon_grouped-tagged',
           'decode_handler': 'SignifiedOnlyFeatureHandler'}, ['vectors__composer']

    # best counting models compared
    yield {'labelled': 'amazon_grouped-tagged',
           'vectors__unlabelled__in': ['wiki', 'wikipedia'],
           'vectors__use_ppmi': True,
           'vectors__algorithm__in': ['count_dependencies', 'count_windows']}, None

    # adding noise vector
    yield {'vectors__composer__in': ['Add'],
           'vectors__unlabelled_percentage__in': ['100.0'],
           'decode_handler__in': ['SignifiedOnlyFeatureHandler'],
           'vectors__unlabelled__in': ['wikipedia', 'wiki'],
           'k__in': ['3'],
           'vectors__algorithm__in': ['word2vec'],
           'labelled__in': ['amazon_grouped-tagged'],
           'document_features__in': ['AN_NN'],
           'vectors__rep__in': ['0']}, ['noise']

    # todo also yield a file name, sequential numbering is very hard to read
    # some more plots in an IPython notebook


for i, (query_dict, fields) in enumerate(get_interesting_experiments()):
    default_fields = ['vectors__algorithm', 'vectors__composer']
    exp_ids = Experiment.objects.values_list('id', flat=True).filter(**query_dict)
    print(exp_ids)
    get_demsar_diagram(*get_demsar_params(exp_ids, name_format=fields if fields else default_fields),
                       filename='demsar%d.pdf' % i)