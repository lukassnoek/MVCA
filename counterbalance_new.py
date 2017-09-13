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
        self.checked_possible = False

    def check_possible(self):

        if self.c_type == 'categorical':

            self.subsample_idx = self._subsample_categorical()

        elif self.c_type == 'continuous':

            self.subsample_idx = self._subsample_continuous()

        else:
            raise ValueError("Please use ctype='continuous' or "
                             "ctype='categorical'")

        self.checked_possible = True
        new_N = float(len(self.subsample_idx))
        print("Size of y after subsampling: %i (%.1f percent reduction in "
              "samples)" % (new_N, (self.y.size - new_N) / self.y.size * 100))

    def _subsample_continuous(self):

        # First, let's do a t-test to check for differences between
        # c | y=0 and c | y=1; thus, only binary c for now
        tstat, pval = ttest_ind(self.c[self.y == 0], self.c[self.y == 1])
        print("Initial pval = %.3f" % pval)
        c_y0 = self.c[self.y == 0]
        c_y1 = self.c[self.y == 1]

        select_idx = np.arange(self.y.size)
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

            if c_y1.size == 1 or c_y0.size == 1:
                # not possible!
                raise ValueError("Not possible to subsample!")

            tstat, pval = ttest_ind(c_y0, c_y1)

        print("Subsampled until p(c | y=0 != p(c | y=1)) > 0.05")
        return np.sort(np.concatenate((idx_c_y0, idx_c_y1)))

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

    def _check_counterbalancing(self, y, c):

        if self.c_type == 'continuous':
            tval, pval = ttest_ind(c[y == 0], c[y == 1])
            return pval < 0.05

        if self.c_type == 'categorical':

            bincounts = np.zeros((np.unique(y).size, np.unique(c).size))
            for i, y_class in enumerate(np.unique(y)):
                bincounts[i, :] = np.bincount(c[y == y_class])

            counterbalanced = np.all(np.all(bincounts == bincounts[0, :],
                                            axis=1))
            return not counterbalanced

    def find_counterbalanced_seed(self, max_attempts=50000):
        """ Find a seed of Stratified K-Fold that gives counterbalanced
        classes """

        if not self.checked_possible:
            self.check_possible()

        if self.c_type == 'categorical':
            raise(IOError('You don''t need to find a seed to get doubly stratified classes. Just use .split()'))
        
        X_tmp = self.X[self.subsample_idx]
        y_tmp = self.y[self.subsample_idx]
        c_tmp = self.c[self.subsample_idx]

        lowest_y_count = np.min(np.bincount(y_tmp))

        if lowest_y_count < self.n_splits:
            raise ValueError("You have to few samples of each class of y for "
                             "n_splits=%i; highest number of splits you can "
                             "use is %i" % (self.n_splits, lowest_y_count))

        seeds = np.random.randint(0, high=1e7, size=max_attempts, dtype=int)

        for i, seed in enumerate(seeds):
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                                  random_state=seed)

            bad_split = False
            for (train_idx, test_idx) in skf.split(X_tmp, y_tmp):

                this_y, this_c = y_tmp[train_idx], c_tmp[train_idx]
                bad_split = self._check_counterbalancing(this_y, this_c)

                if bad_split:
                    print("Not a good split")
                    break

            if not bad_split:
                print("Picking seed %i" % seed)
                self.seed = seed
                break

    def split(self):
        
        if self.c_type == 'categorical':
            if not self.checked_possible:
                self.check_possible()
            y_new = y[self.subsample_idx]
            c_new = c[self.subsample_idx]
            X_new = X[self.subsample_idx,:]
            
            # Create y2 to use for StratifiedKFold
            y2 = np.zeros(y_new.shape[0], dtype=np.int)
            i = 0
            for y_class in np.unique(y_new):
                for c_class in np.unique(c_new):
                    y2[(y_new == y_class) & (c_new == c_class)] = i
                    i += 1
                    
            # SKF, split, and return splits
            skf = StratifiedKFold(n_splits=self.n_splits)
            splits = skf.split(X=X_new, y=y2)

            for (train_idx, test_idx) in splits:
                yield ((self.subsample_idx[train_idx],
                        self.subsample_idx[test_idx]))

if __name__ == '__main__':

    n_samp = 10
    n_feat = 5
    n_fold = 5

    n_half = int(n_samp / 2)
    y = np.repeat([0, 1], repeats=n_half)
    c = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) + np.random.randn(10) / 10000
    data = np.random.randn(n_samp, n_feat)
    data[y == 1, :] += 5.5
    X = data

    # Test continuous
#     css = CounterbalancedStratifiedSplit(X, y, c, n_splits=3,
#                                          c_type='continuous')
#     css.check_possible()
#     css.find_counterbalanced_seed()

    # Test categorical
    y = np.random.binomial(n=1, p=.5, size=100)
    c = np.random.binomial(n=2, p=.5, size=100)
    X = np.random.normal(0, 1, (100, 5))
    splits = CounterbalancedStratifiedSplit(X=X, y=y, c=c, c_type='categorical').split()
    
    for (train_idx, test_idx) in splits:
        print('\nNew fold...')
        for i, set_ in enumerate([train_idx, test_idx]):
            print(['Train set:', 'Test set:'][i])
            y_set = y[set_]
            c_set = c[set_]

            for y_class in np.unique(y_set):
                print('Bincount for y_class: %d' % y_class)
                idx = y_set == y_class
                c_sub = c_set[idx]
                print(np.bincount(c_sub))        