class CounterbalancedStratifiedSplitRandom(object):

    def __init__(self, X, y, c, n_splits=5, c_type='categorical',
                 metric='corr', use_pval=False, threshold=0.05, verbose=False):

        self.X = X
        self.y = y
        self.c = c
        self.z = None
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
                #print(pearsonr(self.y[self.subsample_idx], self.c[self.subsample_idx]))
                return pval > self.threshold
            else:
                return np.abs(stat) < self.threshold

        elif self.c_type == 'categorical':
            bincounts = np.zeros((np.unique(y).size, np.unique(c).size))
            for i, y_class in enumerate(np.unique(y)):
                bincounts[i, :] = np.bincount(c[y == y_class])

            counterbalanced = np.all(bincounts[0, :] == bincounts[1, :])
            return counterbalanced

    def _subsample_continuous(self, iteration=0):

        # First, let's do a t-test to check for differences between
        # c | y=0 and c | y=1; thus, only binary c for now
        self.subsample_idx = np.arange(self.y.size)
        amount = int(1 + np.floor(iteration / 10000))
        this_c = self.c[self.subsample_idx]
        this_y = self.y[self.subsample_idx]

        c_y0 = this_c[this_y == 0]
        c_y1 = this_c[this_y == 1]
        idx_c_y0 = self.subsample_idx[this_y == 0]
        idx_c_y1 = self.subsample_idx[this_y == 1]

        idx_c_y0 = np.random.choice(idx_c_y0, size=idx_c_y0.size - amount, replace=False)
        idx_c_y1 = np.random.choice(idx_c_y1, size=idx_c_y1.size - amount, replace=False)
        self.subsample_idx = np.sort(np.concatenate((idx_c_y0, idx_c_y1)))

        #if iteration % 100 == 0:
        #    print(pearsonr(self.y[self.subsample_idx], self.c[self.subsample_idx]))

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
        # subsample, which is done below:
        final_idx = []
        for i, y_class in enumerate(y_unique):

            this_idx = self.subsample_idx[self.y == y_class]
            this_c = self.c[self.y == y_class]

            for ii, c in enumerate(c_unique):
                final_idx.append(np.random.choice(this_idx[this_c == c],
                                                  int(min_counts[ii]),
                                                  replace=False))

        # The concatenated indices now represent the indices needed to
        # properly subsample the data to make it counterbalanced
        self.subsample_idx = np.sort(np.concatenate(final_idx))

    def _subsample(self, iteration):

        if self.c_type == 'continuous':
            self._subsample_continuous(iteration=iteration)
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

    def _find_counterbalanced_seed(self, max_attempts=10):
        """ Find a seed of Stratified K-Fold that gives counterbalanced
        classes """

        y_tmp = self.y[self.subsample_idx]
        c_tmp = self.c[self.subsample_idx]
        X_tmp = self.X[self.subsample_idx]

        to_stratify = y_tmp if self.z is None else self.z
        if self.c_type == 'categorical':
            lowest_strat_count = np.min(np.bincount(to_stratify))

            if lowest_strat_count < self.n_splits:
                raise ValueError("You have too few samples of each c-y "
                                 "combination to completely counterbalance all "
                                 "your folds with n_splits=%i; highest number of "
                                 "splits you can use is %i" % (self.n_splits,
                                                               lowest_strat_count))

        seeds = np.random.randint(0, high=1e7, size=max_attempts, dtype=int)

        for i, seed in enumerate(seeds):

            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                                  random_state=seed)

            for (train_idx, test_idx) in skf.split(X_tmp, y=to_stratify):

                this_y, this_c = y_tmp[train_idx], c_tmp[train_idx]
                good_split = self._validate_fold(this_y, this_c)
                if not good_split:
                    break

            if good_split:

                if self.verbose:
                    print("Picking seed %i" % seed)

                self.seed = seed
                return True

        return False

    def check_counterbalance_and_subsample(self):

        self.subsample_idx = np.arange(self.y.size)

        if self.c_type == 'continuous':
            found_split = self._find_counterbalanced_seed()
            i = 0
            while not found_split:
                self._subsample(iteration=i)
                found_split = self._find_counterbalanced_seed()
                i += 1
        elif self.c_type == 'categorical':
            self._subsample()
            recode_dict = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
            this_c = self.c[self.subsample_idx]
            this_y = self.y[self.subsample_idx]
            self.z = [recode_dict[(yi, ci)] for yi, ci in zip(this_c, this_y)]
            found_split = self._find_counterbalanced_seed()

        if self.verbose:
            new_N, old_N = len(self.subsample_idx), self.y.size
            print("Size of y after subsampling: %i (%.1f percent reduction in "
                  "samples)" % (new_N, (old_N - new_N) / old_N * 100))

    def split(self, X, y, groups=None):
        """ The final idx to output are subsamples of the subsample_idx... """

        if self.seed is None:
            raise ValueError("Run '.check_counterbalance_and_subsample' "
                             "before you run '.split'!")

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                              random_state=self.seed)

        to_stratify = y if self.z is None else self.z
        for (train_idx, test_idx) in skf.split(X=X, y=to_stratify):
            yield ((train_idx, test_idx))
