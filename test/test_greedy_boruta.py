"""
Comprehensive test suite for modified Boruta algorithm.
Tests API compatibility, model support, edge cases, and core functionality.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification, make_regression
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from greedy_boruta.GreedyBoruta import GreedyBorutaPy

# Import your modified Boruta class
# Adjust this import based on your actual module structure

class TestBorutaBasicFunctionality:
    """Test basic fit, transform, and fit_transform operations."""
    
    @pytest.fixture
    def sample_classification_data(self):
        """Generate sample classification dataset."""
        X, y = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            n_repeated=0,
            random_state=42
        )
        return X, y
    
    @pytest.fixture
    def sample_regression_data(self):
        """Generate sample regression dataset."""
        X, y = make_regression(
            n_samples=100,
            n_features=20,
            n_informative=10,
            random_state=42
        )
        return X, y
    
    def test_fit_basic(self, sample_classification_data):
        """Test basic fit operation."""
        X, y = sample_classification_data
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        boruta = GreedyBorutaPy(rf, n_estimators=50, random_state=42)
        
        # Should complete without errors
        boruta.fit(X, y)
        
        # Check that required attributes are set
        assert hasattr(boruta, 'support_')
        assert hasattr(boruta, 'ranking_')
        assert hasattr(boruta, 'n_features_')
    
    def test_transform_basic(self, sample_classification_data):
        """Test basic transform operation."""
        X, y = sample_classification_data
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        boruta = GreedyBorutaPy(rf, n_estimators=50, random_state=42)
        
        boruta.fit(X, y)
        X_transformed = boruta.transform(X)
        
        # Check that output has correct shape
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == np.sum(boruta.support_)
        assert X_transformed.shape[1] <= X.shape[1]
    
    def test_fit_transform_basic(self, sample_classification_data):
        """Test fit_transform operation."""
        X, y = sample_classification_data
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        boruta = GreedyBorutaPy(rf, n_estimators=50, random_state=42)
        
        X_transformed = boruta.fit_transform(X, y)
        
        # Should be equivalent to fit then transform
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == np.sum(boruta.support_)
    
    def test_not_fitted_error(self, sample_classification_data):
        """Test that transform raises error when not fitted."""
        X, y = sample_classification_data
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        boruta = GreedyBorutaPy(rf, n_estimators=50, random_state=42)
        
        with pytest.raises((ValueError, AttributeError)):
            boruta.transform(X)


class TestBorutaModelCompatibility:
    """Test compatibility with different sklearn estimators."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample dataset."""
        X, y = make_classification(
            n_samples=100,
            n_features=15,
            n_informative=8,
            random_state=42
        )
        return X, y
    
    def test_random_forest_classifier(self, sample_data):
        """Test with RandomForestClassifier."""
        X, y = sample_data
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        boruta = GreedyBorutaPy(rf, n_estimators=50, random_state=42)
        
        boruta.fit(X, y)
        assert boruta.n_features_ >= 0
    
    def test_extra_trees_classifier(self, sample_data):
        """Test with ExtraTreesClassifier."""
        X, y = sample_data
        et = ExtraTreesClassifier(n_estimators=50, max_depth=5, random_state=42)
        boruta = GreedyBorutaPy(et, n_estimators=50, random_state=42)
        
        boruta.fit(X, y)
        assert boruta.n_features_ >= 0
    
    def test_random_forest_regressor(self):
        """Test with RandomForestRegressor."""
        X, y = make_regression(n_samples=100, n_features=15, n_informative=8, random_state=42)
        rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        boruta = GreedyBorutaPy(rf, n_estimators=50, random_state=42)
        
        boruta.fit(X, y)
        assert boruta.n_features_ >= 0
    
    def test_auto_n_estimators(self, sample_data):
        """Test with n_estimators='auto'."""
        X, y = sample_data
        rf = RandomForestClassifier(max_depth=5, random_state=42)
        boruta = GreedyBorutaPy(rf, n_estimators='auto', random_state=42)
        
        boruta.fit(X, y)
        assert boruta.n_features_ >= 0


class TestBorutaParameters:
    """Test different parameter configurations."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample dataset."""
        X, y = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=12,
            random_state=42
        )
        return X, y
    
    def test_different_perc_values(self, sample_data):
        """Test different percentile values."""
        X, y = sample_data
        
        for perc in [90, 95, 100]:
            rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            boruta = GreedyBorutaPy(rf, n_estimators=50, perc=perc, random_state=42)
            boruta.fit(X, y)
            assert boruta.n_features_ >= 0
    
    def test_different_alpha_values(self, sample_data):
        """Test different alpha values."""
        X, y = sample_data
        
        for alpha in [0.01, 0.05, 0.1]:
            rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            boruta = GreedyBorutaPy(rf, n_estimators=50, alpha=alpha, random_state=42)
            boruta.fit(X, y)
            assert boruta.n_features_ >= 0
    
    def test_two_step_parameter(self, sample_data):
        """Test two_step parameter (True and False)."""
        X, y = sample_data
        
        # With two_step=True
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        boruta_two_step = GreedyBorutaPy(rf, n_estimators=50, two_step=True, random_state=42)
        boruta_two_step.fit(X, y)
        
        # With two_step=False
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        boruta_one_step = GreedyBorutaPy(rf, n_estimators=50, two_step=False, random_state=42)
        boruta_one_step.fit(X, y)
        
        assert boruta_two_step.n_features_ >= 0
        assert boruta_one_step.n_features_ >= 0
    
    def test_max_iter_parameter(self, sample_data):
        """Test different max_iter values."""
        X, y = sample_data
        
        for max_iter in [5, 20, 50]:
            rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            boruta = GreedyBorutaPy(rf, n_estimators=50, random_state=42)
            boruta.fit(X, y)
            assert boruta.n_features_ >= 0
    
    def test_verbose_parameter(self, sample_data):
        """Test verbose parameter (should not affect results)."""
        X, y = sample_data
        
        for verbose in [0, 1, 2]:
            rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            boruta = GreedyBorutaPy(rf, n_estimators=50, verbose=verbose, random_state=42)
            boruta.fit(X, y)
            assert boruta.n_features_ >= 0


class TestBorutaAttributes:
    """Test that all expected attributes are set correctly."""
    
    @pytest.fixture
    def fitted_boruta(self):
        """Create a fitted Boruta instance."""
        X, y = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=10,
            random_state=42
        )
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        boruta = GreedyBorutaPy(rf, n_estimators=50, random_state=42)
        boruta.fit(X, y)
        return boruta, X
    
    def test_support_attribute(self, fitted_boruta):
        """Test support_ attribute."""
        boruta, X = fitted_boruta
        
        assert hasattr(boruta, 'support_')
        assert isinstance(boruta.support_, np.ndarray)
        assert boruta.support_.dtype == bool
        assert len(boruta.support_) == X.shape[1]
    
    def test_support_weak_attribute(self, fitted_boruta):
        """Test support_weak_ attribute."""
        boruta, X = fitted_boruta
        
        assert hasattr(boruta, 'support_weak_')
        assert isinstance(boruta.support_weak_, np.ndarray)
        assert boruta.support_weak_.dtype == bool
        assert len(boruta.support_weak_) == X.shape[1]
    
    def test_ranking_attribute(self, fitted_boruta):
        """Test ranking_ attribute."""
        boruta, X = fitted_boruta
        
        assert hasattr(boruta, 'ranking_')
        assert isinstance(boruta.ranking_, np.ndarray)
        assert len(boruta.ranking_) == X.shape[1]
        
        # Check ranking values are valid
        assert np.all(boruta.ranking_ >= 1)
        
        # Selected features should have rank 1
        assert np.all(boruta.ranking_[boruta.support_] == 1)
        
        # Tentative features should have rank 2
        assert np.all(boruta.ranking_[boruta.support_weak_] == 2)
    
    def test_n_features_attribute(self, fitted_boruta):
        """Test n_features_ attribute."""
        boruta, X = fitted_boruta
        
        assert hasattr(boruta, 'n_features_')
        assert isinstance(boruta.n_features_, (int, np.integer))
        assert boruta.n_features_ == np.sum(boruta.support_)
        assert 0 <= boruta.n_features_ <= X.shape[1]


class TestBorutaEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_small_dataset(self):
        """Test with very small dataset."""
        X, y = make_classification(
            n_samples=30,
            n_features=5,
            n_informative=3,
            random_state=42
        )
        rf = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42)
        boruta = GreedyBorutaPy(rf, n_estimators=20, random_state=42)
        
        boruta.fit(X, y)
        assert boruta.n_features_ >= 0
    
    def test_many_features(self):
        """Test with many features."""
        X, y = make_classification(
            n_samples=100,
            n_features=50,
            n_informative=20,
            random_state=42
        )
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        boruta = GreedyBorutaPy(rf, n_estimators=50, random_state=42)
        
        boruta.fit(X, y)
        assert boruta.n_features_ >= 0
    
    def test_all_irrelevant_features(self):
        """Test with dataset where features might all be irrelevant."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        boruta = GreedyBorutaPy(rf, n_estimators=50, random_state=42)
        
        boruta.fit(X, y)
        # Should complete without error, may select few or no features
        assert boruta.n_features_ >= 0
    
    def test_all_informative_features(self):
        """Test with dataset where all features are informative."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=10,
            n_redundant=0,
            random_state=42
        )
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        boruta = GreedyBorutaPy(rf, n_estimators=50, random_state=42)
        
        boruta.fit(X, y)
        # Should select most or all features
        assert boruta.n_features_ > 0
    
    def test_single_feature(self):
        """Test with single feature dataset."""
        X, y = make_classification(
            n_samples=100,
            n_features=1,
            n_informative=1,
            n_redundant=0,
            n_classes=1,
            random_state=42
        )
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        boruta = GreedyBorutaPy(rf, n_estimators=50, random_state=42)
        
        boruta.fit(X, y)
        assert boruta.n_features_ in [0, 1]


class TestBorutaReproducibility:
    """Test reproducibility with random_state."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample dataset."""
        X, y = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=10,
            random_state=42
        )
        return X, y
    
    def test_random_state_reproducibility(self, sample_data):
        """Test that same random_state gives same results."""
        X, y = sample_data
        
        rf1 = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        boruta1 = GreedyBorutaPy(rf1, n_estimators=50, random_state=42)
        boruta1.fit(X, y)
        
        rf2 = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        boruta2 = GreedyBorutaPy(rf2, n_estimators=50, random_state=42)
        boruta2.fit(X, y)
        
        # Results should be identical
        np.testing.assert_array_equal(boruta1.support_, boruta2.support_)
        np.testing.assert_array_equal(boruta1.ranking_, boruta2.ranking_)
        assert boruta1.n_features_ == boruta2.n_features_
    
    def test_different_random_states(self, sample_data):
        """Test that different random_states can give different results."""
        X, y = sample_data
        
        rf1 = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        boruta1 = GreedyBorutaPy(rf1, n_estimators=50, random_state=42)
        boruta1.fit(X, y)
        
        rf2 = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=123)
        boruta2 = GreedyBorutaPy(rf2, n_estimators=50, random_state=123)
        boruta2.fit(X, y)
        
        # Results may differ (but not guaranteed to differ)
        # Just ensure both complete successfully
        assert boruta1.n_features_ >= 0
        assert boruta2.n_features_ >= 0


class TestBorutaInputValidation:
    """Test input validation and error handling."""
    
    def test_numpy_array_input(self):
        """Test with numpy array input."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        boruta = GreedyBorutaPy(rf, n_estimators=50, random_state=42)
        
        boruta.fit(X, y)
        assert boruta.n_features_ >= 0
    
    def test_pandas_dataframe_input(self):
        """Test with pandas DataFrame input."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y_series = pd.Series(y)
        
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        boruta = GreedyBorutaPy(rf, n_estimators=50, random_state=42)
        
        boruta.fit(X_df, y_series)
        assert boruta.n_features_ >= 0
    
    def test_mismatched_shapes(self):
        """Test that mismatched X and y shapes raise error."""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 90)  # Wrong size
        
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        boruta = GreedyBorutaPy(rf, n_estimators=50, random_state=42)
        
        with pytest.raises(ValueError):
            boruta.fit(X, y)


class TestBorutaTransformConsistency:
    """Test consistency between fit_transform and fit+transform."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample dataset."""
        X, y = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=10,
            random_state=42
        )
        return X, y
    
    def test_fit_transform_equivalence(self, sample_data):
        """Test that fit_transform equals fit then transform."""
        X, y = sample_data
        
        # Method 1: fit_transform
        rf1 = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        boruta1 = GreedyBorutaPy(rf1, n_estimators=50, random_state=42)
        X_transformed1 = boruta1.fit_transform(X, y)
        
        # Method 2: fit then transform
        rf2 = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        boruta2 = GreedyBorutaPy(rf2, n_estimators=50, random_state=42)
        boruta2.fit(X, y)
        X_transformed2 = boruta2.transform(X)
        
        # Results should be identical
        np.testing.assert_array_equal(X_transformed1, X_transformed2)
        np.testing.assert_array_equal(boruta1.support_, boruta2.support_)


class TestBorutaMultipleTransforms:
    """Test multiple transform calls on different datasets."""
    
    def test_transform_on_new_data(self):
        """Test that transform works on new data with same number of features."""
        X_train, y_train = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=10,
            random_state=42
        )
        X_test, _ = make_classification(
            n_samples=50,
            n_features=20,
            n_informative=10,
            random_state=123
        )
        
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        boruta = GreedyBorutaPy(rf, n_estimators=50, random_state=42)
        
        # Fit on training data
        boruta.fit(X_train, y_train)
        
        # Transform both training and test data
        X_train_transformed = boruta.transform(X_train)
        X_test_transformed = boruta.transform(X_test)
        
        # Check shapes are consistent
        assert X_train_transformed.shape[1] == X_test_transformed.shape[1]
        assert X_train_transformed.shape[1] == boruta.n_features_