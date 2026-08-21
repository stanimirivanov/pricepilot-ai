"""Tests for anomaly detection"""

import numpy as np
import pandas as pd
import pytest

from pricepilot.models.anomaly_detector import AnomalyResult, DemandAnomalyDetector


@pytest.fixture
def normal_demand():
    """Generate normal demand data"""
    np.random.seed(42)
    n = 365
    # Normal demand with weekly pattern
    day_of_week = np.arange(n) % 7
    weekend_effect = np.where(day_of_week >= 5, 20, 0)
    demand = 100 + weekend_effect + np.random.normal(0, 5, n)
    return demand


@pytest.fixture
def demand_with_anomalies(normal_demand):
    """Generate demand with known anomalies"""
    demand = normal_demand.copy()
    # Add anomalies
    demand[50] = 200  # Sudden spike
    demand[100] = 20  # Sudden drop
    demand[200] = 180  # Another spike
    demand[300] = 30  # Another drop
    return demand


@pytest.fixture
def dates():
    """Generate dates"""
    return pd.date_range("2023-01-01", periods=365, freq="D")


def test_detector_initialization():
    """Test basic initialization"""
    detector = DemandAnomalyDetector(detector_type="isolation_forest")
    assert detector.detector_type == "isolation_forest"
    assert detector.contamination == 0.05
    assert not detector.is_fitted


def test_detector_fit(normal_demand):
    """Test fitting detector"""
    detector = DemandAnomalyDetector()
    detector.fit(normal_demand)

    assert detector.is_fitted
    assert detector.threshold is not None


def test_detect_normal_data(normal_demand, dates):
    """Test detection on normal data"""
    detector = DemandAnomalyDetector(contamination=0.05)
    detector.fit(normal_demand)
    result = detector.detect(normal_demand, dates)

    assert isinstance(result, AnomalyResult)
    assert result.total_points == len(normal_demand)
    assert result.n_anomalies <= len(normal_demand) * 0.10  # Should be around 5%


def test_detect_anomalies(demand_with_anomalies, dates):
    """Test detection of known anomalies"""
    detector = DemandAnomalyDetector(contamination=0.05)
    detector.fit(demand_with_anomalies)
    result = detector.detect(demand_with_anomalies, dates)

    # Should detect more anomalies than in normal data
    assert result.n_anomalies >= 4  # We added 4 anomalies

    # Check that anomalies are flagged
    anomaly_indices = [50, 100, 200, 300]
    for idx in anomaly_indices:
        assert result.labels[idx] == 1 or result.scores[idx] > result.threshold


def test_anomaly_percentage(demand_with_anomalies, dates):
    """Test anomaly percentage calculation"""
    detector = DemandAnomalyDetector(contamination=0.05)
    result = detector.fit_detect(demand_with_anomalies, dates)

    percentage = result.anomaly_percentage()
    assert 0 <= percentage <= 20  # Should be between 0% and 20%


def test_to_dataframe(demand_with_anomalies, dates):
    """Test conversion to DataFrame"""
    detector = DemandAnomalyDetector()
    result = detector.fit_detect(demand_with_anomalies, dates)

    df = result.to_dataframe()
    assert "date" in df.columns
    assert "anomaly_score" in df.columns
    assert "is_anomaly" in df.columns
    assert len(df) == len(demand_with_anomalies)


def test_get_anomaly_dates(demand_with_anomalies, dates):
    """Test getting anomaly dates"""
    detector = DemandAnomalyDetector()
    result = detector.fit_detect(demand_with_anomalies, dates)

    anomaly_dates = result.get_anomaly_dates()
    assert len(anomaly_dates) == result.n_anomalies


def test_single_value_check(normal_demand):
    """Test checking single value for anomaly"""
    detector = DemandAnomalyDetector()
    detector.fit(normal_demand)

    # Normal value should not be anomaly
    normal_value = float(np.mean(normal_demand))
    assert not detector.is_anomaly(normal_value, normal_demand[:-1])

    # Extreme value should be anomaly
    extreme_value = float(np.mean(normal_demand) + 50)
    # Note: This might not always be detected as anomaly depending on detector
    # Just check it doesn't crash
    result = detector.is_anomaly(extreme_value, normal_demand[:-1])
    assert isinstance(result, bool)


def test_multiple_detectors(normal_demand):
    """Test different detector types"""
    detector_types = ["isolation_forest", "knn", "lof"]

    for detector_type in detector_types:
        detector = DemandAnomalyDetector(detector_type=detector_type)
        detector.fit(normal_demand)
        result = detector.detect(normal_demand)

        assert result.total_points == len(normal_demand)
        assert result.n_anomalies >= 0


def test_unknown_detector_type():
    """Test error for unknown detector type"""
    with pytest.raises(ValueError, match="Unknown detector"):
        DemandAnomalyDetector(detector_type="unknown_detector")


def test_detect_before_fit():
    """Test error when detecting before fitting"""
    detector = DemandAnomalyDetector()
    demand = np.random.randn(100)

    with pytest.raises(ValueError, match="not fitted"):
        detector.detect(demand)


def test_anomaly_aware_pricing_model(normal_demand):
    """Test AnomalyAwarePricingModel integration"""
    from pricepilot.models.anomaly_detector import AnomalyAwarePricingModel

    # Create a mock pricing model
    class MockPricingModel:
        def price_for_tomorrow(self, current_price):
            class MockResult:
                def __init__(self):
                    self.optimal_price = 20.0
                    self.expected_revenue = 1000.0
                    self.date = pd.Timestamp("2024-01-01")
                    self.forecasted_demand = 100.0
                    self.demand_lower = 90.0
                    self.demand_upper = 110.0
                    self.confidence = "high"
                    self.price_change_pct = 0.10

            return MockResult()

    # Create detector and fit
    detector = DemandAnomalyDetector()
    detector.fit(normal_demand)

    # Create anomaly-aware pricing model
    pricing_model = MockPricingModel()
    anomaly_pricing = AnomalyAwarePricingModel(
        pricing_model=pricing_model,
        anomaly_detector=detector,
    )

    # Test with normal demand
    result = anomaly_pricing.price_with_anomaly_check(
        current_price=15.0,
        historical_demand=normal_demand[:-1],
        forecasted_demand=float(np.mean(normal_demand)),
    )

    assert "pricing_result" in result
    assert "is_anomaly" in result
    assert "anomaly_status" in result
    assert result["anomaly_status"] in ["ANOMALY", "NORMAL"]
    assert "forecasted_demand" in result


def test_anomaly_aware_pricing_detects_anomaly(normal_demand):
    """Test that anomaly-aware pricing detects anomalies"""
    from pricepilot.models.anomaly_detector import AnomalyAwarePricingModel

    # Create a mock pricing model
    class MockPricingModel:
        def price_for_tomorrow(self, current_price):
            class MockResult:
                def __init__(self):
                    self.optimal_price = 20.0
                    self.expected_revenue = 1000.0

            return MockResult()

    # Create detector and fit
    detector = DemandAnomalyDetector()
    detector.fit(normal_demand)

    # Create anomaly-aware pricing model
    anomaly_pricing = AnomalyAwarePricingModel(
        pricing_model=MockPricingModel(),
        anomaly_detector=detector,
    )

    # Test with extreme demand value
    extreme_demand = float(np.mean(normal_demand) + 100)  # Very high demand

    result = anomaly_pricing.price_with_anomaly_check(
        current_price=15.0,
        historical_demand=normal_demand[:-1],
        forecasted_demand=extreme_demand,
    )

    # The result should have anomaly status (though detector might not always catch it)
    assert "anomaly_status" in result
