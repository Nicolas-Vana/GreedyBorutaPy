# Greedy Boruta

[![License](https://img.shields.io/github/license/Nicolas-Vana/GreedyBorutaPy)](https://github.com/Nicolas-Vana/GreedyBorutaPy/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/GreedyBoruta.svg)](https://badge.fury.io/py/GreedyBoruta)
![Test Coverage](./coverage.svg)

A faster variant of the [Boruta all-relevant feature selection method](https://www.jstatsoft.org/article/view/v036i11) with **greedy feature confirmation** that achieves **5-40x speedups** through a confirmation criterion relaxation.

This implementation is a fork of [`boruta_py`](https://github.com/scikit-learn-contrib/boruta_py), with modifications focused on **improving computational efficiency while maintaining statistical rigor**.

**[Read the full article explaining the algorithm and experimental results](LINK_TO_BE_ADDED)**

## Greedy Confirmation

Unlike the vanilla Boruta algorithm, which requires features to achieve statistical significance through binomial testing before confirmation, **Greedy Boruta confirms any feature that beats the maximum shadow importance at least once**. This simple change leads to:

- **5-40x faster convergence** on tested datasets.
- **Automatic determination of `max_iter`** based on the significance level $\alpha$. In practice, no manual tuning is needed.
- **Equal or higher recall** compared to vanilla Boruta. We prove that Greedy Boruta does not miss relevant features that are identified by its vanilla counterpart.
- **Guaranteed convergence** in $O(-log_2 \alpha)$ iterations

The algorithm automatically calculates the minimum iterations needed for a feature with zero hits to be rejected as $log_2(1/\alpha)$, then runs until all features are confirmed or rejected (which occurs at or before this limit).

## How to Install

Install with `pip`:

```shell
pip install greedyboruta
```

or with `conda`:

```shell
conda install -c conda-forge greedyboruta
```

## Dependencies

* numpy
* scipy
* scikit-learn

## How to Use

The interface is identical to `scikit-learn` and `boruta_py`:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from greedyboruta import GreedyBorutaPy

# load X and y
X = pd.read_csv('examples/test_X.csv', index_col=0).values
y = pd.read_csv('examples/test_y.csv', header=None, index_col=0).values
y = y.ravel()

# define random forest classifier
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

# define GreedyBoruta feature selection method
# max_iter is automatically determined based on alpha
feat_selector = GreedyBorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

# find all relevant features - typically 5-40x faster than standard Boruta
feat_selector.fit(X, y)

# check selected features
feat_selector.support_

# check ranking of features
feat_selector.ranking_

# transform X to selected features
X_filtered = feat_selector.transform(X)
```

## Philosophy: All-relevant vs minimal-optimal

Greedy Boruta, like the vanilla Boruta, follows the **all-relevant** feature selection philosophy. This means it aims to find **every feature that carries useful information**, not just the smallest set that achieves good prediction.

**Why this matters:**
- When you want to **understand a phenomenon** (not just predict it), you need all contributing factors.
- In **scientific discovery** and **causal inference**, missing a relevant feature can lead to incorrect conclusions.
- **Redundant features** (correlated with informative ones) are intentionally retained - they carry signal even if not strictly necessary.
- Downstream **minimal-optimal methods** (RFE, LASSO, mRMR) can further reduce the feature set if needed.

This philosophy justifies the greedy confirmation criterion: in all-relevant selection, **false negatives (missing relevant features) are more costly than false positives (including a few extra irrelevant features)**. The relaxed criterion prioritizes high recall, which aligns perfectly with the all-relevant paradigm.

## What's different from vanilla Boruta?

### Core algorithm change

**Greedy confirmation criterion**: features are confirmed immediately upon achieving **at least one hit** (beating the maximum shadow importance in any iteration) rather than requiring statistical significance through binomial testing. The rejection criterion remains unchanged. In practice, this change:

1. **Maintains or improves recall**: any feature confirmed by vanilla Boruta will also be confirmed by Greedy Boruta (since statistical significance requires at least one hit).
2. **Enables guaranteed convergence**: until the last iteration, all tentative features have exactly zero hits, simplifying the rejection test.
3. **Dramatically speeds up and parallelization**: Greedy Boruta runs at most $K$ iterations, which is the same number of iterations at which the vanilla Boruta confirms or rejects its "first batch" of features. Unlike the vanilla Boruta, however, Greedy Boruta can be parallelize as $K$ is known since it exclusively depends on the significance level $\alpha$.
4. **Trades slight specificity for speed**: the reduction in specificity by the relaxation of the confirmation criterion is relatively small when compared to the speed gains.

### Automatic `max_iter` Calculation

Because all tentative features have exactly zero hits (since confirmed features have at least one), the binomial test for rejection is nicely simplified. The algorithm computes the minimum number of iterations needed for a feature with zero hits to be rejected at significance level $\alpha$.

For a binomial test with $p = 0.5$ and $x = 0$ hits,
```
p-value = (1/2)^n < alpha.
```
Therefore, **`max_iter` = $O(log_2(1/\alpha))$**.

This means that all features will be sorted into confirmed or rejected in at most `max_iter` iterations - at this iteration, all remaining tentative features (with zero hits) are automatically rejected, and all features with hits > 0 are confirmed. No statistical testing is required during intermediate iterations.

**With False discovery rate (FDR) correction applied (as in `boruta_py`), `max_iter` values are:**
- $\alpha = 0.10$: ~6 iterations
- $\alpha = 0.01$: ~10 iterations  
- $\alpha = 0.001$: ~14 iterations
- $\alpha = 0.0001$: ~18 iterations
- $\alpha = 0.0001$: ~22 iterations

**Notice that given a significance level, no manual tuning of `max_iter` is required!**

## What's inherited from `boruta_py`?

This implementation builds upon the excellent work in `boruta_py` and retains all its key improvements over the original `R` implementation:

* **Faster run times** thanks to `scikit-learn`
* **Scikit-learn interface**: `fit`, `transform`, `fit_transform`, etc.
* **Compatible with any ensemble method** from `scikit-learn`.
* **Automatic n_estimator selection**.
* **Feature ranking**.
* **Percentile threshold** (`perc` parameter) for more flexible shadow feature comparison.
* **Two-step correction** (FDR + Bonferroni) for multiple testing.

We highly recommend using pruned trees with depth between 3-7, as suggested in the original `boruta_py` documentation.

## Parameters

**`estimator`** : object
   > A supervised learning estimator with a 'fit' method that returns the
   > feature_importances_ attribute. Important features must correspond to
   > high absolute values in the feature_importances_.

**`n_estimators`** : int or string, default = 1000
   > If int, sets the number of estimators in the chosen ensemble method.
   > If 'auto', this is determined automatically based on dataset size.

**`perc`** : int, default = 100
   > Percentile of shadow feature importances to use as threshold.
   > The default (100) uses the maximum, equivalent to vanilla Boruta.
   > Lower values (e.g., 90) are less stringent and may select more features.

**`alpha`** : float, default = 0.05
   > Significance level for the corrected p-values in both correction steps.
   > Also automatically determines max_iter via the formula: log_2(1/alpha).
   > Lower alpha = more conservative selection = more iterations.

**`two_step`** : Boolean, default = True
   > If True, uses FDR + Bonferroni correction. If False, uses only
   > Bonferroni correction (original Boruta behavior with perc=100).

**`random_state`** : int, RandomState instance or None, default = None
   > Random seed for reproducibility.

**`verbose`** : int, default = 0
   > Controls verbosity of output:
   > 0 = silent, 1 = iteration counter, 2 = detailed statistics per iteration

### Removed parameters

Unlike the vanilla Boruta implementation, Greedy Boruta does **not** require:
- **`max_iter`**: Automatically calculated from `alpha`.
- **`early_stopping`**: Not needed due to guaranteed convergence.
- **`n_iter_no_change`**: Not needed due to guaranteed convergence.

This simplification improves usability and eliminates the need for manual tuning of convergence-related parameters.

## Attributes

**`n_features_`** : int
   > The number of selected features (confirmed only).

**`support_`** : array of shape [n_features]
   > Boolean mask of selected features (confirmed features only).

**`support_weak_`** : array of shape [n_features]
   > Boolean mask of tentative features that didn't gain enough support.

**`ranking_`** : array of shape [n_features]
   > Feature ranking where confirmed features = 1, tentative features = 2,
   > and rejected features have ranks ≥ 3 based on importance.

**`importance_history_`** : array of shape [n_iterations, n_features]
   > Historical record of feature importances across all iterations.

## Performance comparison: Greedy Boruta vs vanilla Boruta

Based on synthetic experiments with known ground truth, we observed

- **5-15x speedup** on challenging datasets with proper early stopping for vanilla Boruta.
- **Up to 40x speedup** when vanilla Boruta runs without early stopping to full convergence.
- **Equal or higher recall**, as Greedy Boruta _never_ misses features that the vanilla Boruta would find relevant.
- **Slightly lower specificity**: $<10$ features selected on 500-feature datasets tested).
- **Guaranteed convergence**: all features are always classified (no tentative features remain).

## When to use Greedy Boruta

**Use Greedy Boruta when:**
- You want **all-relevant feature selection** with high recall.
- You're working with **high-dimensional data** for which the vanilla Boruta takes too long to run/converge.
- **Computational efficiency matters**.
- False positives can be filtered in **downstream pipelines** using regularization, cross-validation, minimal-optimal selection.
- You want to avoid manually tuning `max_iter` or early stopping parameters.

**Consider standard Boruta when:**
- You need **maximum specificity** and false positives are very costly.
- Your dataset is **small enough** that speed isn't a concern.
- **Statistical conservatism** is paramount for your application.

## References

1. COLOCAR REFERENCIA DO GREEDY BORUTA AQUI!
2. Kursa M., Rudnicki W., "Feature Selection with the Boruta Package" Journal of Statistical Software, Vol. 36, Issue 11, Sep 2010
3. Homola D., "BorutaPy: An all-relevant feature selection method" https://github.com/scikit-learn-contrib/boruta_py

## Credits

This implementation is built upon [`boruta_py`](https://github.com/scikit-learn-contrib/boruta_py) by Daniel Homola, which itself is based on the original Boruta algorithm by Miron B. Kursa and Witold R. Rudnicki.

The greedy confirmation criterion and automatic convergence calculation are novel contributions of this fork, based on findings by Nicolas Vana Santos and Estevão Batista do Prado.

## Citation

If you use Greedy Boruta in your research, please cite the Greedy Boruta paper and both the original Boruta paper and the `boruta_py` implementation:

```
@article{greedyboruta2025,
  title={The Greedy Boruta Algorithm: Faster Feature Selection Without SacrificingRecall},
  author={BLA BLA BLA},
  journal={BLA BLA},
  volume={XXX},
  number={XXX},
  pages={XXX},
  year={XXX}
}

@article{kursa2010feature,
  title={Feature selection with the Boruta package},
  author={Kursa, Miron B and Rudnicki, Witold R},
  journal={Journal of Statistical Software},
  volume={36},
  number={11},
  pages={1--13},
  year={2010}
}
```

## License

This project maintains the same BSD-3-Clause license as `boruta_py`.
