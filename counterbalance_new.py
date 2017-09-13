import numpy as np
from scipy.stats import ttest_ind
from sklearn.model_selection import StratifiedKFold


class CounterbalancedStratifiedSplit(object):

    def __init__(self, X, y, c, n_splits=5,
                 c_type='categorical'):
        self.X = X
        self.y = y
        self.c = c
        self.n_splits = n_splits
        self.c_type = c_type

    def check_possible(self):

        if self.c_type == 'categorical':

            self.subsample_idx = self._check_categorical()

        elif self.c_type == 'continuous':

            self.subsample_idx = self._check_continuous()

        else:
            raise ValueError("Please use ctype='continuous' or "
                             "ctype='categorical'")

        self.checked_possible = True

    def _check_continuous(self):

        # First, let's do a t-test to check for differences between
        # c | y=0 and c | y=1; thus, only binary c for now
        tstat, pval = ttest_ind(self.c[self.y == 0], self.c[self.y == 1])

        c_y0 = self.c[self.y == 0]
        c_y1 = self.c[self.y == 1]

        select_idx = self.subsample_idx.copy()
        idx_c_y0 = select_idx[self.y == 0]
        idx_c_y1 = select_idx[self.y == 1]

        # Remove the samples influencing the t-test most until p > 0.05
        while pval < 0.05 or c_y0.size < 3:

            to_drop_c0 = c_y0.argmax() if tstat > 0 else c_y0.argmin()
            idx_c_y0 = np.delete(idx_c_y0, to_drop_c0)
            c_y0 = np.delete(c_y0, to_drop_c0)

            to_drop_c1 = c_y1.argmax() if tstat < 0 else c_y0.argmin()
            idx_c_y1 = np.delete(idx_c_y1, to_drop_c1)
            c_y1 = np.delete(c_y1, to_drop_c1)

            if not c_y1 or not c_y0:
                # not possible!
                raise ValueError("Not possible to subsample!")

            tstat, pval = ttest_ind(c_y0, c_y1)

        print("Subsampled until p(c | y=0 != p(c | y=1)) > 0.05")
        return np.sort(np.concatenate((idx_c_y0, idx_c_y1)))

    def _check_categorical(self):

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
        subsample_idx = np.arange(self.y.size)
        final_idx = []
        for i, y_class in enumerate(y_unique):

            this_idx = subsample_idx[self.y == y_class]
            this_c = self.c[self.y == y_class]

            for ii, c in enumerate(c_unique):
                final_idx.append(np.random.choice(this_idx[this_c == c],
                                                  int(min_counts[ii]),
                                                  replace=False))

        final_idx = np.sort(np.concatenate(final_idx))

        # Just to check!
        bincounts = np.zeros((y_unique.size, c_unique.size))
        for i, y_class in enumerate(y_unique):
            this_c = self.c[final_idx[self.y[final_idx] == y_class]]
            bincounts[i, :] = np.bincount(this_c)

        assert(np.all(np.all(bincounts == bincounts[0,:], axis=1)))
        return final_idx

    def find_counterbalanced_seed(self, max_attempts=50000):
        """ Find a seed of Stratified K-Fold that gives counterbalanced
        classes """

        if not self.checked_possible:
            self.check_possible()



if __name__ == '__main__':

    n_samp = 10
    n_feat = 5
    n_fold = 5

    n_half = int(n_samp / 2)
    y = np.repeat([0, 1], repeats=n_half)
    c = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1])
    data = np.random.randn(n_samp, n_feat)
    data[y == 1, :] += 5.5
    X = data

    css = CounterbalancedStratifiedSplit(X, y, c, n_splits=5,
                                         c_type='categorical')
    css.check_possible()
