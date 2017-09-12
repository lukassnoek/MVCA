import numpy as np
from scipy.stats import itemfreq
from sklearn.model_selection import StratifiedKFold


class CounterbalancedStratifiedSplit(object):

    def __init__(self, X, y, c, counterbalance_tolerance=.05, n_splits=5,
                 verbose=False, c_type='categorical'):
        self.X = X
        self.y = y
        self.c = c
        self.counterbalance_tolerance = counterbalance_tolerance
        self.n_splits = n_splits
        self.verbose = verbose
        self.c_type = c_type

        self.subsample_idx = np.arange(len(y))
        self.y_classes = np.unique(y)

        if c_type == 'categorical':
            self.c_classes = np.unique(c)
            self.n_confound_classes = len(self.c_classes)
            self.counterbalanced_proportion = 1.0 / self.n_confound_classes

        self._seed = None
        self.checked_possible = False

    def _check_categorical(self, y_class, idx, select_idx):

        freqs = itemfreq(self.c[idx])

        # Get absolute difference, diffs = distance-matrix
        diffs = freqs[:, 1].reshape(-1, 1) - freqs[:, 1]

        # If any absolute difference is larger than 0, we have an imbalance
        if np.max(diffs) > 0:
            print('Oops! The confound classes are not counterbalanced '
                  'within y-class %s. Subsampling...' % str(y_class))

            for c_class in self.c_classes:
                # From self.subsample_idx, sample the idx of the number of
                # y-class observations necessary to obtain a balanced class
                isect = (self.y == y_class) & (self.c == c_class)
                randsub = np.random.choice(self.subsample_idx[isect],
                                           size=np.min(freqs[:, 1]),
                                           replace=False)
                select_idx = np.concatenate((select_idx, randsub))
        else:
            # If confound classes are counterbalanced within this y_class,
            # select all of these idx
            select_idx = np.concatenate((select_idx,
                                         self.subsample_idx[idx]))

        return select_idx

    def _check_continuous(self):

        from scipy.stats import ttest_ind
        tstat, pval = ttest_ind(self.c[self.y == 0], self.c[self.y == 1])

        c_y0 = self.c[self.y == 0]
        c_y1 = self.c[self.y == 1]
        select_idx = self.subsample_idx.copy()
        idx_c_y0 = select_idx[self.y == 0]
        idx_c_y1 = select_idx[self.y == 1]

        while pval < 0.05 or c_y0.size < 3:

            to_drop_c0 = c_y0.argmax() if tstat > 0 else c_y0.argmin()
            idx_c_y0 = np.delete(idx_c_y0, to_drop_c0)
            c_y0 = np.delete(c_y0, to_drop_c0)

            to_drop_c1 = c_y1.argmax() if tstat < 0 else c_y0.argmin()
            idx_c_y1 = np.delete(idx_c_y1, to_drop_c1)
            c_y1 = np.delete(c_y1, to_drop_c1)
            tstat, pval = ttest_ind(c_y0, c_y1)

            if pval < 0.05:
                success = True

        return np.sort(np.concatenate((idx_c_y0, idx_c_y1))), pval

    def check_possible(self):
        """ Check if the confound classes are counterbalanced in each class of
        y. If not, a subsampling index will be calculated such that they are.
        """

        if self.verbose:
            print('Checking if a counterbalanced split is possible without '
                  'subsampling...')

        if self.c_type == 'continuous':
            select_idx, pval = self._check_continuous()

            if self.verbose:
                print("Final p-value = %.3f (%i samples)" % (pval, select_idx.size))

        elif self.c_type == 'categorical':
            select_idx = np.array([])
            for y_class in self.y_classes:
                idx = self.y == y_class
                select_idx = self._check_categorical(y_class, idx, select_idx)

            if self.verbose:
                print('After subsampling, proportions of c-classes in y_classes:')
                y_tmp = self.y[self.subsample_idx]
                c_tmp = self.c[self.subsample_idx]

                for y_class in self.y_classes:
                    print(itemfreq(c_tmp[y_tmp == y_class]))

        self.checked_possible = True
        self.subsample_idx = np.sort(select_idx).astype(int)


    def find_counterbalanced_seed(self, max_attempts=50000):
        """ Find a seed of Stratified K-Fold that gives counterbalanced
        classes """

        if not self.checked_possible:
            self.check_possible()

        X_tmp = self.X.copy()[self.subsample_idx]
        y_tmp = self.y.copy()[self.subsample_idx]
        c_tmp = self.c.copy()[self.subsample_idx]

        seeds = np.random.randint(0, high=1e7, size=max_attempts, dtype=int)
        for i, seed in enumerate(seeds):
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                                  random_state=seed)
            bad_split = False
            for (train_idx, test_idx) in skf.split(X_tmp, y_tmp):

                if bad_split:
                    if self.verbose and i % 100 == 0:
                        print('.', end='')
                    break

                for set_ in (train_idx, test_idx):
                    y_set = y_tmp[set_]
                    c_set = c_tmp[set_]



                    for y_class in self.y_classes:
                        c_set_class = c_set[y_set == y_class]

                        if self.c_type == 'categorical':
                            bad_split = self._check_fold_c_categorical(c_set_class)

            if not bad_split:
                if self.verbose:
                    print('\nGood split found! Seed %d was used.' % i)
                self.seed = seed
                break

        if self.seed is None:
            raise(ValueError('\nSorry, could not find any good split...'))

    def _check_fold_c_continuous():
        pass

    def _check_fold_c_categorical(self, c_set_class):
        # Get proportions of each class of c for each y_class
        freqs = itemfreq(c_set_class).astype(float)[:, 1]
        proportions = freqs / len(c_set_class)

        # Check whether the proportions are equal
        diffs = proportions.reshape(-1, 1) - proportions
        maxdiffs = np.max(diffs)
        reject = maxdiffs > self.counterbalance_tolerance
        if proportions.shape[0] == 1 or reject:
            return True
        else:
            return False

    def split(self, X, y):
        """ The final idx to output are subsamples of the subsample_idx... """

        if self.seed is None:
            raise(ValueError('You need to run CounterbalancedStratifiedSplit.'
                             'find_counterbalanced_seed() first'))

        X_tmp = X[self.subsample_idx]
        y_tmp = y[self.subsample_idx]

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                              random_state=self.seed)

        for (train_idx, test_idx) in skf.split(X=X_tmp, y=y_tmp):
            yield ((self.subsample_idx[train_idx],
                    self.subsample_idx[test_idx]))


if __name__ == '__main__':

    n_samp = 100
    n_feat = 5
    n_fold = 5

    n_half = int(n_samp / 2)
    y = np.repeat([0, 1], repeats=n_half)
    c = np.roll(y, 10) + np.random.randn(y.size)

    data = np.random.randn(n_samp, n_feat)
    data[c == 1, :] += 5
    data[y == 1, :] += 5.5
    X = data

    css = CounterbalancedStratifiedSplit(X, y, c, n_splits=5, verbose=True,
                                         counterbalance_tolerance=.05,
                                         c_type='continuous')
    css.check_possible()
    y_tmp = y[css.subsample_idx]
    c_tmp = c[css.subsample_idx]

    css.find_counterbalanced_seed()
    splts = css.split(X, y)
