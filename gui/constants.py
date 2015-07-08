METRIC_DB = 'accuracy'
METRIC_CSV_FILE = 'accuracy_score'
BOOTSTRAP_REPS = 500
SIGNIFICANCE_LEVEL = 0.01
CLASSIFIER = 'MultinomialNB'

# some names are too long to fit in figures, abbreviate as follows:
ABBREVIATIONS = {
    'word2vec': 'w2v',
    # 'turian': 'Tur',
    'count_windows': 'W',
    'count_dependencies': 'D',
    'random_neigh': 'RandN',
    'random_vect': 'RandV',
    # 'Observed': 'obs',
    'amazon_grouped-tagged': 'AM',
    'reuters21578/r8-tagged-grouped': 'R2',
    'movie-reviews-tagged': 'MR',
    'techtc100-clean/Exp_186330_94142-tagged': 'TTC1',
    'techtc100-clean/Exp_22294_25575-tagged': 'TTC2',
    'techtc100-clean/Exp_324745_85489-tagged': 'TTC3',
    'techtc100-clean/Exp_47456_497201-tagged': 'TTC4',
    'techtc100-clean/Exp_69753_85489-tagged': 'TTC5'
}