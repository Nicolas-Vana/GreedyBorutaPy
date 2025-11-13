# Boruta Test Suite

This test suite ensures your modified Boruta algorithm maintains API compatibility with the original boruta_py library.

## Test Files

### 1. `test_boruta.py` - Comprehensive Test Suite
Full pytest-based test suite with 50+ test cases covering:

- **Basic Functionality** (TestBorutaBasicFunctionality)
  - `fit()`, `transform()`, and `fit_transform()` operations
  - NotFittedError handling
  
- **Model Compatibility** (TestBorutaModelCompatibility)
  - RandomForestClassifier
  - ExtraTreesClassifier
  - RandomForestRegressor
  - n_estimators='auto' functionality

- **Parameter Testing** (TestBorutaParameters)
  - Different `perc` values (90, 95, 100)
  - Different `alpha` values (0.01, 0.05, 0.1)
  - `two_step` parameter (True/False)
  - `max_iter` variations
  - `verbose` levels

- **Attribute Verification** (TestBorutaAttributes)
  - `support_` - boolean mask of selected features
  - `support_weak_` - boolean mask of tentative features
  - `ranking_` - feature ranking array
  - `n_features_` - count of selected features

- **Edge Cases** (TestBorutaEdgeCases)
  - Small datasets
  - Large feature sets
  - All irrelevant features
  - All informative features
  - Single feature datasets

- **Reproducibility** (TestBorutaReproducibility)
  - random_state consistency
  - Different random states

- **Input Validation** (TestBorutaInputValidation)
  - NumPy arrays
  - Pandas DataFrames
  - Mismatched shapes

- **Transform Consistency** (TestBorutaTransformConsistency)
  - fit_transform vs fit+transform equivalence
  - Multiple transforms on different data

### 2. `test_boruta_smoke.py` - Quick Smoke Tests
Fast, essential tests for rapid development feedback:
- Basic fit operation
- Basic transform operation
- fit_transform method
- Attribute checks
- Reproducibility
- Different model types
- Parameter variations

## Running the Tests

### Quick Smoke Tests (Recommended for Development)
```bash
python test_boruta_smoke.py
```
Takes ~10-20 seconds. Perfect for quick validation during development.

### Full Test Suite
```bash
# Install pytest if needed
pip install pytest

# Run all tests with verbose output
pytest test_boruta.py -v

# Run specific test class
pytest test_boruta.py::TestBorutaBasicFunctionality -v

# Run with coverage
pip install pytest-covc:\Users\Administrador\Documents\Code\GreedyBorutaPy
pytest test_boruta.py --cov=boruta --cov-report=html
```

### Run Both Test Suites
```bash
# Quick smoke tests first
python test_boruta_smoke.py && pytest test_boruta.py -v
```

## Expected Behavior

Your modified Boruta should:

1. **Maintain API Compatibility**
   - Accept same parameters as original BorutaPy
   - Implement fit(), transform(), fit_transform()
   - Follow sklearn transformer interface

2. **Set Required Attributes After Fitting**
   - `support_`: Boolean array marking confirmed features
   - `support_weak_`: Boolean array marking tentative features
   - `ranking_`: Integer array with feature rankings (1=selected, 2=tentative, 3+=rejected)
   - `n_features_`: Integer count of selected features

3. **Handle Edge Cases**
   - Small datasets
   - Many features
   - All/no informative features
   - Different data types (numpy, pandas)

4. **Support Different Estimators**
   - Any sklearn estimator with `fit()` and `feature_importances_` attribute
   - RandomForest, ExtraTrees, GradientBoosting, etc.
   - Both classifiers and regressors

## Customizing Tests

### Adding New Tests
Add to `test_boruta.py`:

```python
def test_my_new_feature(self, sample_data):
    """Test description."""
    X, y = sample_data
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    boruta = BorutaPy(rf, n_estimators=50, max_iter=10, random_state=42)
    
    boruta.fit(X, y)
    
    # Your assertions here
    assert something_is_true
```

### Adjusting Test Parameters
If tests are too slow, reduce:
- `n_estimators` (default: 50)
- `max_iter` (default: 10)
- Dataset sizes

If tests are too lenient, increase:
- Dataset complexity
- Number of test iterations
- Assertion strictness

## Common Issues and Solutions

### ImportError: cannot import name 'BorutaPy'
Update the import statement in both test files:
```python
# Change this line based on your module structure
from your_module import BorutaPy  # or whatever your import path is
```

### Tests Running Slowly
Smoke tests should complete in 10-20 seconds. Full suite may take 2-5 minutes.
To speed up:
```python
# Reduce n_estimators
rf = RandomForestClassifier(n_estimators=20, ...)  # instead of 50

# Reduce max_iter
boruta = BorutaPy(..., max_iter=5, ...)  # instead of 10
```

### Flaky Tests
Some tests may occasionally fail due to randomness. If you see inconsistent failures:
1. Check if `random_state` is set consistently
2. Increase `max_iter` for more stable results
3. Use larger datasets for more reliable importance estimates

## Test Coverage Goals

Aim for:
- ✅ 100% of public API methods tested
- ✅ All parameters tested with valid values
- ✅ Edge cases covered
- ✅ Error conditions tested
- ✅ Integration with sklearn estimators verified

## Additional Testing Recommendations

### Performance Testing
```python
import time

X, y = make_classification(n_samples=1000, n_features=100, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
boruta = BorutaPy(rf, n_estimators=100, max_iter=20, random_state=42)

start = time.time()
boruta.fit(X, y)
elapsed = time.time() - start

print(f"Fitted in {elapsed:.2f} seconds")
print(f"Selected {boruta.n_features_} features")
```

### Comparing with Original Boruta
```python
from boruta import BorutaPy as BorutaModified
from boruta_original import BorutaPy as BorutaOriginal

X, y = make_classification(n_samples=100, n_features=20, random_state=42)
rf1 = RandomForestClassifier(n_estimators=50, random_state=42)
rf2 = RandomForestClassifier(n_estimators=50, random_state=42)

modified = BorutaModified(rf1, n_estimators=50, max_iter=10, random_state=42)
original = BorutaOriginal(rf2, n_estimators=50, max_iter=10, random_state=42)

modified.fit(X, y)
original.fit(X, y)

print(f"Modified selected: {modified.n_features_}")
print(f"Original selected: {original.n_features_}")
```

## Contributing

When adding new features to your Boruta modification:
1. Add tests first (TDD approach)
2. Run smoke tests frequently
3. Run full test suite before committing
4. Update this README if API changes

## Questions?

If tests fail unexpectedly:
1. Check that your modifications maintain the same API
2. Verify all required attributes are set after fitting
3. Ensure random_state is handled consistently
4. Review the specific test that failed for clues

Happy testing! 🚀
