from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DataGeneratorConfig:
    """Configuration for synthetic data generation"""

    start_date: str = "2022-01-01"
    end_date: str = "2023-12-31"
    base_price: float = 15.0
    price_elasticity: float = -2.0
    weather_sensitivity: float = 10.0
    weekend_multiplier: float = 1.3
    noise_std: float = 5.0
    seed: int = 42


class CarWashDataGenerator:
    """Generate synthetic car wash transaction data"""

    def __init__(self, config: DataGeneratorConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def generate_weather(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate synthetic weather data"""
        # Simple weather model: more rain in winter, less in summer
        month = dates.month
        is_winter = (month >= 11) | (month <= 2)

        # Rain probability
        rain_prob = np.where(is_winter, 0.4, 0.2)
        is_raining = self.rng.random(len(dates)) < rain_prob

        # Temperature (Fahrenheit)
        base_temp = 70 + 20 * np.sin((month - 3) * 2 * np.pi / 12)
        temp_noise = self.rng.normal(0, 5, len(dates))
        temperature = base_temp + temp_noise

        return pd.DataFrame(
            {
                "date": dates,
                "is_raining": is_raining,
                "temperature": temperature,
                "is_sunny": ~is_raining & (temperature > 60),
            }
        )

    def generate_prices(self, n_days: int) -> np.ndarray:
        """Generate price history with some variation"""
        base_price = self.config.base_price
        # Random walk around base price
        price_changes = self.rng.normal(0, 0.5, n_days)
        prices = base_price + np.cumsum(price_changes)
        # Clamp to reasonable range
        prices = np.clip(prices, base_price * 0.7, base_price * 1.3)
        return prices

    def calculate_demand(self, prices: np.ndarray, weather: pd.DataFrame) -> np.ndarray:
        """Calculate demand based on price and weather"""
        # Base demand
        demand = 100.0

        # Price effect
        price_effect = self.config.price_elasticity * (prices - self.config.base_price)

        # Weather effect
        weather_effect = self.config.weather_sensitivity * weather["is_sunny"].values

        # Day of week effect
        day_of_week = weather["date"].dt.dayofweek.values
        weekend_effect = np.where(
            day_of_week >= 5,  # Saturday=5, Sunday=6
            self.config.weekend_multiplier * 10,
            0,
        )

        # Combine effects
        demand = demand + price_effect + weather_effect + weekend_effect

        # Add noise
        noise = self.rng.normal(0, self.config.noise_std, len(prices))
        demand = demand + noise

        # Ensure non-negative
        demand = np.maximum(demand, 0)

        return demand

    def generate(self) -> pd.DataFrame:
        """Generate complete synthetic dataset"""
        # Create date range
        dates = pd.date_range(start=self.config.start_date, end=self.config.end_date, freq="D")

        # Generate components
        weather = self.generate_weather(dates)
        prices = self.generate_prices(len(dates))
        demand = self.calculate_demand(prices, weather)
        revenue = prices * demand

        # Combine into dataframe
        df = pd.DataFrame(
            {
                "date": dates,
                "price": prices,
                "quantity_sold": demand.astype(int),
                "revenue": revenue,
                "is_raining": weather["is_raining"],
                "is_sunny": weather["is_sunny"],
                "temperature": weather["temperature"],
                "day_of_week": dates.dayofweek,
                "is_weekend": dates.dayofweek >= 5,
                "month": dates.month,
                "year": dates.year,
            }
        )

        return df

    def save(self, df: pd.DataFrame, filepath: str) -> None:
        """Save dataframe to CSV"""
        df.to_csv(filepath, index=False)

    def generate_and_save(self, filepath: str) -> pd.DataFrame:
        """Generate data and save to file"""
        df = self.generate()
        self.save(df, filepath)
        return df
