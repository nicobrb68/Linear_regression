import os
import json
import math
import csv
import sys
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self, data_path: str = "trained.json"):
        self.theta0 = 0.0
        self.theta1 = 0.0
        self.data_path = data_path
        self.kilometers: list[float] = []
        self.kilometer_mini: float = 0.0
        self.kilometer_max: float = 0.0
        self.prices: list[float] = []
        self.price_mini: float = 0.0
        self.price_max: float = 0.0

    def _normalize(self, data: list[float]) -> tuple[list[float],
                                                     float, float]:
        min_val: float = min(data)
        max_val: float = max(data)
        max_range: float = max_val - min_val
        try:
            if max_range == 0:
                raise ValueError("Zero division avoided")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        normalized: list[float] = []
        for x in data:
            normalized.append((x - min_val) / (max_range))
        return normalized, min_val, max_val

    def hypothese(self, mileage: float) -> float:
        result = self.theta0 + (self.theta1 * mileage)
        return result

    def load(self) -> None:
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path) as f:
                    raw_result: dict = json.load(f)
                    self.theta0, self.theta1 = (float(raw_result["theta0"]),
                                                float(raw_result["theta1"]))
                    if (not math.isfinite(self.theta0)
                            or not math.isfinite(self.theta1)):
                        raise ValueError("Invalid Data")
            except (OSError, json.JSONDecodeError, KeyError, ValueError,
                    TypeError) as e:
                print(f"Error: {e}")
                self.theta0 = 0.0
                self.theta1 = 0.0

    def load_dataset(self, csv_path: str = "data.csv") -> None:
        self.kilometers = []
        self.prices = []
        try:
            with open(csv_path) as f:
                line = csv.DictReader(f)
                for row in line:
                    km = float(row["km"])
                    price = float(row["price"])
                    if (not math.isfinite(price)
                            or not math.isfinite(km) or km < 0 or price < 0):
                        raise ValueError("Invalid Data")
                    self.kilometers.append(km)
                    self.prices.append(price)
                if len(self.kilometers) < 2:
                    raise ValueError("Data too short too generate a result")
        except (OSError, csv.Error, KeyError, ValueError,
                TypeError) as e:
            print(f"Error: {e}")
            sys.exit(1)

    def train(self, learning_rate: float = 0.1, iteration: int = 3000) -> None:
        if not self.kilometers:
            sys.exit(1)
        (norm_km, self.kilometer_mini,
         self.kilometer_max) = self._normalize(self.kilometers)
        (norm_price, self.price_mini,
         self.price_max) = self._normalize(self.prices)
        total_car = len(norm_km)
        self.theta0 = 0.0
        self.theta1 = 0.0
        for _ in range(iteration):
            sum_error_0: float = 0.0
            sum_error_1: float = 0.0
            for i in range(total_car):
                prediction = self.hypothese(norm_km[i])
                error: float = prediction - norm_price[i]
                sum_error_0 += error
                sum_error_1 += error * norm_km[i]
            tmp_theta0 = (self.theta0 -
                          (learning_rate * (sum_error_0 / total_car)))
            tmp_theta1 = (self.theta1 -
                          (learning_rate * (sum_error_1 / total_car)))
            self.theta0 = tmp_theta0
            self.theta1 = tmp_theta1
        self._denormalize()

    def _denormalize(self) -> None:
        price_range: float = (self.price_max - self.price_mini)
        km_range: float = (self.kilometer_max - self.kilometer_mini)
        real_theta1 = self.theta1 * (price_range / km_range)
        real_theta0 = (self.price_mini + (self.theta0 * price_range) -
                       (real_theta1 * self.kilometer_mini))
        self.theta1 = real_theta1
        self.theta0 = real_theta0

    def save(self) -> None:
        try:
            if (not math.isfinite(self.theta0) or
                    not math.isfinite(self.theta1)):
                raise ValueError("Invalid Data")
            with open(self.data_path, "w") as f:
                json.dump({"theta0": self.theta0, "theta1": self.theta1},
                          f, indent=4)
        except (OSError, ValueError, TypeError) as e:
            print(f"Error: {e}")
            sys.exit(1)

    def plot(self) -> None:
        if not self.kilometers:
            sys.exit(1)
        plt.scatter(self.kilometers, self.prices, label="Real Data")
        x_line = [self.kilometer_mini, self.kilometer_max]
        y_line = [
            self.hypothese(self.kilometer_mini),
            self.hypothese(self.kilometer_max),
        ]
        plt.plot(x_line, y_line, color="red", label="Linear Regression")
        plt.xlabel("Kilometer (km)")
        plt.ylabel("Price (€)")
        plt.title("Car Price about their Kilometers")
        plt.legend()
        plt.show()

    def precision(self) -> float:
        if not self.kilometers:
            sys.exit(1)
        moy_price: float = sum(self.prices) / len(self.prices)
        sum_error: float = 0.0
        sum_tot: float = 0.0
        for i in range(len(self.prices)):
            sum_error += (
                (self.prices[i] - self.hypothese(self.kilometers[i])) ** 2
            )
            sum_tot += (self.prices[i] - moy_price) ** 2
        if sum_tot == 0:
            return 0.0
        return 1 - (sum_error / sum_tot)
