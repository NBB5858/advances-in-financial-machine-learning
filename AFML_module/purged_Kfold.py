import numpy as np
from sklearn.model_selection._split import _BaseKFold

class PurgedKFold(_BaseKFold):
    '''
    Extend KFold class to work with labels that span intervals
    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False)
    '''

    def __init__(self, n_splits=3, embargo_size=0):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.n_splits = n_splits
        self.embargo_size = embargo_size

    def split(self, obs_frame):
        '''
        obs_frame should have columns: ['event start', 'barrier', 'absolute return', 'first touch',
                                        'event observation time', 'label observation time', 'feature']
        and have a consecutive index starting at 0.

        Returns test and train indices of the observation frame
        '''

        # test_ranges are indices of the observation frame
        test_ranges = [(i[0], i[-1]) for i in np.array_split(np.arange(obs_frame.shape[0]), self.n_splits)]
        for i, j in test_ranges:

            test_start_time = obs_frame["event observation time"].iloc[i]
            test_end_time = obs_frame["label observation time"].iloc[j]

            left_train_frame = obs_frame[(obs_frame["label observation time"] < test_start_time)]

            if j < obs_frame.shape[0] - 1:
                right_train_frame = obs_frame[obs_frame["event start"] > obs_frame["first touch"].iloc[j] + self.embargo_size]
                train_indices = np.concat([left_train_frame.index, right_train_frame.index])

            else:  # no right frame, test set is at end of dataset
                train_indices = left_train_frame.index

            test_indices = np.arange(i, j + 1)

            yield train_indices, test_indices