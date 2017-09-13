import numpy as np
from scipy.stats import ttest_ind, pearsonr
from sklearn.model_selection import StratifiedKFold


class CounterbalancedStratifiedSplit(object):

    def __init__(self, X, y, c, n_splits=5,
                 c_type='categorical', metric='corr', use_pval=False,
                 threshold=0.05, verbose=False):
        self.X = X
        self.y = y
        self.c = c
        self.n_splits = n_splits
        self.c_type = c_type
        self.metric = metric
        self.use_pval = use_pval
        self.threshold = threshold
        self.seed = None
        self.verbose = verbose

    def _validate_fold(self, y, c):

        if self.c_type == 'continuous':

            if self.metric == 'corr':
                stat, pval = pearsonr(c, y)
            elif self.metric == 'tstat':
                stat, pval = ttest_ind(c[y == 0], c[y == 1])
            else:
                raise ValueError("Please choose either 'corr' or 'tstat'!")

            if self.use_pval:
                return pval > self.threshold
            else:
                return np.abs(stat) < self.threshold

        elif self.c_type == 'categorical':

            bincounts = np.zeros((np.unique(y).size, np.unique(c).size))
            for i, y_class in enumerate(np.unique(y)):
                bincounts[i, :] = np.bincount(c[y == y_class])

            counterbalanced = np.all(bincounts[0, :] == bincounts[1, :])
            return counterbalanced

    def _subsample_continuous(self):

        # First, let's do a t-test to check for differences between
        # c | y=0 and c | y=1; thus, only binary c for now
        this_c = self.c[self.subsample_idx]
        this_y = self.y[self.subsample_idx]

        if self.metric == 'tstat':
            stat, pval = ttest_ind(this_c[this_y == 0], this_c[this_y == 1])
            stat *= -1  # To make it consistent with corr
        else:
            stat, pval = pearsonr(this_c, this_y)

        if self.verbose:
            stat_word = 'correlation' if self.metric == 'corr' else 'tstat'
            print("Current overall %s c/y (pval): "
                  "%.3f (%.3f)" % (stat_word, stat, pval))

        c_y0 = this_c[this_y == 0]
        c_y1 = this_c[this_y == 1]
        idx_c_y0 = self.subsample_idx[this_y == 0]
        idx_c_y1 = self.subsample_idx[this_y == 1]

        to_drop_c0 = c_y0.argmax() if stat < 0 else c_y0.argmin()
        idx_c_y0 = np.delete(idx_c_y0, to_drop_c0)
        c_y0 = np.delete(c_y0, to_drop_c0)

        to_drop_c1 = c_y1.argmax() if stat > 0 else c_y0.argmin()
        idx_c_y1 = np.delete(idx_c_y1, to_drop_c1)
        c_y1 = np.delete(c_y1, to_drop_c1)

        self.subsample_idx = np.sort(np.concatenate((idx_c_y0, idx_c_y1)))

    def _subsample_categorical(self):

        c_unique = np.unique(self.c)
        y_unique = np.unique(self.y)
        counts = np.zeros((y_unique.size, c_unique.size))

        # Count how many times a c appears for a given y
        # so, count(c | y_i)
        for i, y_class in enumerate(y_unique):
            this_c = self.c[self.y == y_class]
            counts[i, :] = np.array([(c == this_c).sum() for c in c_unique])

        # ... yielding a len(y_unique) x len(c_unique) matrix
        # Now, take the minimum across rows
        min_counts = counts.min(axis=0)

        if np.all(min_counts == 0):
            msg = ("Wow, your data is really messed up ... There is no way to "
                   "subsample it, because the minimum proportion of all values"
                   "of c across all values of y is 0 ...")
            raise ValueError(msg)

        # Which are exactly the number of trials (per c) which you need to
        # subsample
        final_idx = []
        for i, y_class in enumerate(y_unique):

            this_idx = self.subsample_idx[self.y == y_class]
            this_c = self.c[self.y == y_class]

            for ii, c in enumerate(c_unique):
                final_idx.append(np.random.choice(this_idx[this_c == c],
                                                  int(min_counts[ii]),
                                                  replace=False))

        self.subsample_idx = np.sort(np.concatenate(final_idx))

    def _subsample(self):

        if self.c_type == 'continuous':
            self._subsample_continuous()
        elif self.c_type == 'categorical':
            self._subsample_categorical()
        else:
            raise ValueError("Please pick c_type='categorical' or "
                             "c_type='continuous'")

        if len(self.subsample_idx) < (2 * len(np.unique(self.y))):
            msg = ("Probably subsampled too much (only have %i samples now); "
                   "this dataset can't be meaningfully "
                   "counterbalanced" % len(self.subsample_idx))
            raise ValueError(msg)

    def _find_counterbalanced_seed(self, max_attempts=1000):
        """ Find a seed of Stratified K-Fold that gives counterbalanced
        classes """

        y_tmp = self.y[self.subsample_idx]
        c_tmp = self.c[self.subsample_idx]
        X_tmp = self.X[self.subsample_idx]
        lowest_y_count = np.min(np.bincount(y_tmp))

        if lowest_y_count < self.n_splits:
            raise ValueError("You have to few samples of each class of y for "
                             "n_splits=%i; highest number of splits you can "
                             "use is %i" % (self.n_splits, lowest_y_count))

        seeds = np.random.randint(0, high=1e7, size=max_attempts, dtype=int)

        for i, seed in enumerate(seeds):
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                                  random_state=seed)

            for (train_idx, test_idx) in skf.split(X_tmp, y_tmp):

                this_y, this_c = y_tmp[train_idx], c_tmp[train_idx]
                good_split = self._validate_fold(this_y, this_c)

                if not good_split:
                    break

            if good_split:
                print("Picking seed %i" % seed)
                self.seed = seed
                return good_split

        return good_split

    def split(self, X, y):
        """ The final idx to output are subsamples of the subsample_idx... """

        self.subsample_idx = np.arange(self.y.size)

        if self.c_type == 'continuous':
            found_split = self._find_counterbalanced_seed()
            while not found_split:
                self._subsample()
                found_split = self._find_counterbalanced_seed()
        elif self.c_type == 'categorical':
            self._subsample()
            found_split = self._find_counterbalanced_seed()

        if self.verbose:
            new_N = float(len(self.subsample_idx))
            old_N = self.y.size
            print("Size of y after subsampling: %i (%.1f percent reduction in "
                  "samples)" % (new_N, (old_N - new_N) / old_N * 100))

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
    c = np.roll(y, 10)
    X = np.random.randn(n_samp, n_feat)

    css = CounterbalancedStratifiedSplit(X, y, c, n_splits=3,
                                         c_type='categorical', verbose=True,
                                         metric='tstat', threshold=1)
    folds = css.split(X, y)

    for train_idx, test_idx in folds:
        this_c = c[train_idx]
        this_y = y[train_idx]
        print(this_c[this_y == 0])
        print(this_c[this_y == 1])
        print(pearsonr(this_c, this_y))
