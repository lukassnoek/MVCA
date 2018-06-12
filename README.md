# How to control for confounds in decoding analyses of neuroimaging data ("MultiVoxel Confound Analysis", MVCA)
Code for simulations and empirical analyses for our [article](https://www.biorxiv.org/content/early/2018/03/28/290684) on dealing with confounds in decoding analyses of neuroimaging data. The data (preprocessed VBM/TBSS arrays) can be downloaded using the `download_data.py` script.

We've implemented the (cross-validated) confound regression procedure as a scikit-learn-style "transformer" class, which is implemented in the [skbold](https://github.com/lukassnoek/skbold) package. To use it, install `skbold` (from master) and import it as follows: `from skbold.preproc import ConfoundRegressor`.

NEW: we also uploaded the source code for the `ConfoundRegressor` object in this repository at analyses/confounds.py,
so no need to install `skbold` (feel free to use/modify/adapt/etc).

A minimal example on how to use it is outlined below:

```python
import numpy as np
from skbold.preproc import ConfoundRegressor

data = np.random.normal(0, 1, size=(100, 2))

# X = neuroimaging data (N x K array), C = confound (N x P array, for P confound variables)
X, C = data[:, 0, np.newaxis], data[:, 1]
X_train, X_test = X[::2, :], X[1::2, :]

cfr = ConfoundRegressor(confound=C, X=X)
X_train_corr = cfr.fit_transform(X_train)
X_test_corr = cfr.transform(X_test)

# X_train_corr and X_test_corr represent the data corrected for the confound
```

The `ConfoundRegressor` class also works with scikit-learn pipelines!

```python
import numpy as np
from skbold.preproc import ConfoundRegressor
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

data = np.random.normal(0, 1, size=(100, 2))

# X = neuroimaging data (N x K array), C = confound (N x P array, for P confound variables)
X, C = data[:, 0, np.newaxis], data[:, 1]
y = np.repeat([0, 1], repeats=50)

cfr = ConfoundRegressor(confound=C, X=X)
clf = SVC(kernel='linear')
pipeline = make_pipeline(cfr, clf)

scores = cross_val_score(estimator=pipeline, X=X, y=y, cv=10)
```
