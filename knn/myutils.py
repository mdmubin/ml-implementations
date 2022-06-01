import numpy as np
from numpy.random import permutation, seed


def split_data(dataset):
    """
    DESCRIPTION
    -----------
    Split the given dataset into a training data, validation data and testing
    data, in the ratio of 7:2:1

    ## EXAMPLE USAGE:

    >>> from myutils import split_data
    >>> data = split_data(raw_data)
    >>> data['train'] # get the split data

    RETURNS
    -------
    A dict containing the split data. Each set of data can be accessed by using
    'train', 'test' or 'validation' subscript on the return value
    """

    DATA_LEN = (dataset['data']).shape[0]

    seed(31)
    shuffled_indices = permutation(DATA_LEN)

    # Starting and ending indices
    TRAIN_START = 0
    TRAIN_END = int(.7 * DATA_LEN)

    VAL_START = TRAIN_END
    VAL_END = TRAIN_END + int(0.2 * DATA_LEN)

    TEST_START = VAL_END
    TEST_END = DATA_LEN

    # Split data
    train_set = np.array([dataset['data'][i]
                          for i in shuffled_indices[TRAIN_START: TRAIN_END]])

    val_set = np.array([dataset['data'][i]
                        for i in shuffled_indices[VAL_START: VAL_END]])

    test_set = np.array([dataset['data'][i]
                        for i in shuffled_indices[TEST_START:TEST_END]])

    # Split targets
    train_target = np.array([dataset['target'][i]
                            for i in shuffled_indices[TRAIN_START: TRAIN_END]])

    val_target = np.array([dataset['target'][i]
                           for i in shuffled_indices[VAL_START: VAL_END]])

    test_target = np.array([dataset['target'][i]
                           for i in shuffled_indices[TEST_START:TEST_END]])

    return {
        'data': {
            'train': train_set,
            'validation': val_set,
            'test': test_set
        },

        'target': {
            'train': train_target,
            'validation': val_target,
            'test': test_target
        }
    }


def load_csv_data(filename):
    from_file = np.genfromtxt(filename, delimiter=',', dtype=np.float64)

    return {
        'data': from_file[:, :-1],
        'target': np.array(from_file[:, -1], dtype=np.int64)
    }
