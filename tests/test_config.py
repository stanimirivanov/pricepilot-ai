from pricepilot.config.settings import Settings


def test_default_settings():
    """Test default settings load correctly"""
    settings = Settings()
    assert settings.environment == "development"
    assert settings.model.min_price < settings.model.max_price
    assert settings.business_rules.break_even_price > 0


def test_yaml_config_loading(tmp_path):
    """Test YAML configuration loading"""
    yaml_content = """
model:
  min_price: 3.0
  max_price: 30.0
"""
    yaml_file = tmp_path / "test_config.yaml"
    yaml_file.write_text(yaml_content)

    settings = Settings.load_yaml(yaml_file)
    assert settings.model.min_price == 3.0
    assert settings.model.max_price == 30.0


def test_business_rules_validation():
    """Test business rules are sensible"""
    settings = Settings()
    rules = settings.business_rules
    assert rules.break_even_price <= settings.model.max_price
    assert 0 <= rules.competitor_weight <= 1
