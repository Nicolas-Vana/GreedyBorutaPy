"""
Quick smoke tests for Boruta - fast tests to catch obvious breakages.
Run this for quick validation during development.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Import your modified Boruta
from GreedyBoruta import GreedyBorutaPy


def test_basic_fit():
    """Quick test that basic fit works."""
    print("Testing basic fit...")
    X, y = make_classification(n_samples=100, n_features=20, n_informative=10, random_state=42)
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    boruta = GreedyBorutaPy(rf, n_estimators=50, max_iter=10, random_state=42)
    
    boruta.fit(X, y)
    
    assert hasattr(boruta, 'support_')
    assert hasattr(boruta, 'ranking_')
    assert hasattr(boruta, 'n_features_')
    print("✓ Basic fit works")


def test_basic_transform():
    """Quick test that transform works."""
    print("Testing basic transform...")
    X, y = make_classification(n_samples=100, n_features=20, n_informative=10, random_state=42)
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    boruta = GreedyBorutaPy(rf, n_estimators=50, max_iter=10, random_state=42)
    
    boruta.fit(X, y)
    X_transformed = boruta.transform(X)
    
    assert X_transformed.shape[0] == X.shape[0]
    assert X_transformed.shape[1] == np.sum(boruta.support_)
    print(f"✓ Transform works - selected {X_transformed.shape[1]}/{X.shape[1]} features")


def test_fit_transform():
    """Quick test that fit_transform works."""
    print("Testing fit_transform...")
    X, y = make_classification(n_samples=100, n_features=20, n_informative=10, random_state=42)
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    boruta = GreedyBorutaPy(rf, n_estimators=50, max_iter=10, random_state=42)
    
    X_transformed = boruta.fit_transform(X, y)
    
    assert X_transformed.shape[0] == X.shape[0]
    assert X_transformed.shape[1] <= X.shape[1]
    print(f"✓ Fit_transform works - selected {X_transformed.shape[1]}/{X.shape[1]} features")


def test_attributes():
    """Quick test that all expected attributes exist."""
    print("Testing attributes...")
    X, y = make_classification(n_samples=100, n_features=20, n_informative=10, random_state=42)
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    boruta = GreedyBorutaPy(rf, n_estimators=50, max_iter=10, random_state=42)
    
    boruta.fit(X, y)
    
    # Check all required attributes
    assert hasattr(boruta, 'support_'), "Missing support_ attribute"
    assert hasattr(boruta, 'support_weak_'), "Missing support_weak_ attribute"
    assert hasattr(boruta, 'ranking_'), "Missing ranking_ attribute"
    assert hasattr(boruta, 'n_features_'), "Missing n_features_ attribute"
    
    # Check attribute types and shapes
    assert isinstance(boruta.support_, np.ndarray), "support_ should be numpy array"
    assert boruta.support_.dtype == bool, "support_ should be boolean array"
    assert len(boruta.support_) == X.shape[1], "support_ has wrong length"
    
    assert isinstance(boruta.ranking_, np.ndarray), "ranking_ should be numpy array"
    assert len(boruta.ranking_) == X.shape[1], "ranking_ has wrong length"
    
    assert isinstance(boruta.n_features_, (int, np.integer)), "n_features_ should be int"
    assert boruta.n_features_ == np.sum(boruta.support_), "n_features_ doesn't match support_"
    
    print("✓ All attributes present and correct")


def test_reproducibility():
    """Quick test that random_state ensures reproducibility."""
    print("Testing reproducibility...")
    X, y = make_classification(n_samples=100, n_features=20, n_informative=10, random_state=42)
    
    rf1 = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    boruta1 = GreedyBorutaPy(rf1, n_estimators=50, max_iter=10, random_state=42)
    boruta1.fit(X, y)
    
    rf2 = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    boruta2 = GreedyBorutaPy(rf2, n_estimators=50, max_iter=10, random_state=42)
    boruta2.fit(X, y)
    
    np.testing.assert_array_equal(boruta1.support_, boruta2.support_)
    np.testing.assert_array_equal(boruta1.ranking_, boruta2.ranking_)
    
    print("✓ Reproducibility works")


def test_different_models():
    """Quick test with different model types."""
    print("Testing different model types...")
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor
    from sklearn.datasets import make_regression
    
    X_clf, y_clf = make_classification(n_samples=100, n_features=15, random_state=42)
    X_reg, y_reg = make_regression(n_samples=100, n_features=15, random_state=42)
    
    # Test ExtraTreesClassifier
    et = ExtraTreesClassifier(n_estimators=50, max_depth=5, random_state=42)
    boruta = GreedyBorutaPy(et, n_estimators=50, max_iter=10, random_state=42)
    boruta.fit(X_clf, y_clf)
    print(f"  ✓ ExtraTreesClassifier works - selected {boruta.n_features_} features")
    
    # Test RandomForestRegressor
    rf_reg = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    boruta = GreedyBorutaPy(rf_reg, n_estimators=50, max_iter=10, random_state=42)
    boruta.fit(X_reg, y_reg)
    print(f"  ✓ RandomForestRegressor works - selected {boruta.n_features_} features")


def test_parameters():
    """Quick test of different parameter combinations."""
    print("Testing different parameters...")
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    
    # Test different perc values
    for perc in [90, 100]:
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        boruta = GreedyBorutaPy(rf, n_estimators=50, perc=perc, max_iter=5, random_state=42)
        boruta.fit(X, y)
    print("  ✓ Different perc values work")
    
    # Test two_step parameter
    for two_step in [True, False]:
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        boruta = GreedyBorutaPy(rf, n_estimators=50, two_step=two_step, max_iter=5, random_state=42)
        boruta.fit(X, y)
    print("  ✓ two_step parameter works")
    
    # Test n_estimators='auto'
    rf = RandomForestClassifier(max_depth=5, random_state=42)
    boruta = GreedyBorutaPy(rf, n_estimators='auto', max_iter=5, random_state=42)
    boruta.fit(X, y)
    print("  ✓ n_estimators='auto' works")


def run_all_smoke_tests():
    """Run all smoke tests."""
    print("=" * 60)
    print("Running Boruta Smoke Tests")
    print("=" * 60)
    
    tests = [
        test_basic_fit,
        test_basic_transform,
        test_fit_transform,
        test_attributes,
        test_reproducibility,
        test_different_models,
        test_parameters
    ]
    
    failed = []
    for test in tests:
        try:
            print()
            test()
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed.append((test.__name__, e))
    
    print()
    print("=" * 60)
    if not failed:
        print("All smoke tests PASSED! ✓")
        print("=" * 60)
        return True
    else:
        print(f"{len(failed)} test(s) FAILED:")
        for name, error in failed:
            print(f"  - {name}: {error}")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = run_all_smoke_tests()
    exit(0 if success else 1)
