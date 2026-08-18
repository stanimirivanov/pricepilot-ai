import pytest

from pricepilot.optimization.price_optimizer import OptimizationConfig, PriceOptimizer


def linear_demand(price: float) -> float:
    """Simple linear demand function: Q = 100 - 2P"""
    return 100 - 2 * price


@pytest.fixture
def optimizer():
    """Create optimizer with linear demand"""
    config = OptimizationConfig(
        min_price=5.0,
        max_price=50.0,
        max_price_change_pct=0.50,  # Allow 50% change for testing
    )
    return PriceOptimizer(linear_demand, config)


def test_optimal_price_linear_demand(optimizer):
    """Test that optimizer finds theoretical optimum"""
    # For linear demand Q = 100 - 2P, revenue R = P(100-2P) = 100P - 2P^2
    # dR/dP = 100 - 4P = 0 => P = 25
    result = optimizer.optimize(current_price=20.0)

    assert result.success
    assert abs(result.optimal_price - 25.0) < 1.0
    assert result.expected_revenue > 0


def test_price_constraints(optimizer):
    """Test that constraints are respected"""
    # Test with very low current price
    result = optimizer.optimize(current_price=10.0)
    assert result.optimal_price >= 5.0  # Min price
    assert result.optimal_price <= 50.0  # Max price

    # Test with very high current price
    result = optimizer.optimize(current_price=45.0)
    assert result.optimal_price >= 5.0
    assert result.optimal_price <= 50.0


def test_price_change_limit(optimizer):
    """Test that price change is limited"""
    config_strict = OptimizationConfig(
        min_price=5.0,
        max_price=50.0,
        max_price_change_pct=0.10,  # Only 10% change allowed
    )
    strict_optimizer = PriceOptimizer(linear_demand, config_strict)

    current_price = 20.0
    result = strict_optimizer.optimize(current_price=current_price)

    max_allowed = current_price * 1.10
    min_allowed = current_price * 0.90

    assert min_allowed <= result.optimal_price <= max_allowed


def test_break_even_constraint(optimizer):
    """Test that price never goes below break-even"""
    config = OptimizationConfig(
        min_price=5.0,
        max_price=50.0,
        break_even_price=15.0,
        max_price_change_pct=1.0,  # Allow large changes
    )
    break_even_optimizer = PriceOptimizer(linear_demand, config)

    result = break_even_optimizer.optimize(current_price=10.0)
    assert result.optimal_price >= 15.0  # Never below break-even


def test_revenue_improvement(optimizer):
    """Test that optimization improves revenue"""
    current_price = 15.0
    current_demand = linear_demand(current_price)
    current_revenue = current_price * current_demand

    result = optimizer.optimize(current_price=current_price)

    assert result.expected_revenue > current_revenue


def test_analysis_output(optimizer):
    """Test price range analysis"""
    analysis = optimizer.analyze_price_range(current_price=20.0)

    assert "prices" in analysis
    assert "demand" in analysis
    assert "revenue" in analysis
    assert len(analysis["prices"]) == 100
    assert len(analysis["demand"]) == 100
    assert len(analysis["revenue"]) == 100
