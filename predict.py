import os
import json
import math
import sys


def fetch_data(data_path: str) -> tuple:
    if os.path.exists(data_path):
        try:
            with open(data_path) as f:
                raw_result: dict = json.load(f)
                result: tuple = (float(raw_result["theta0"]),
                                 float(raw_result["theta1"]))
                if not math.isfinite(result[0]) or not math.isfinite(result[1]):
                    raise ValueError("Invalid Data")
                return result
        except (OSError, json.JSONDecodeError, KeyError, ValueError,
                TypeError) as e:
            print(f"Error: {e}")
            return (0.0, 0.0)
    else:
        return (0.0, 0.0)


def hypothese(mileage: float, theta0: float, theta1: float) -> float:
    result = theta0 + (theta1 * mileage)
    return result


def main() -> None:
    theta0, theta1 = fetch_data("trained.json")
    try:
        print("Insert kilometer:")
        kilometer = float(input())
        if (kilometer < 0 or not math.isfinite(kilometer)):
            raise ValueError("Invalid Mileage")
    except (ValueError, OSError, EOFError, KeyboardInterrupt, TypeError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    result = hypothese(kilometer, theta0, theta1)
    print(result)



if __name__ == "__main__":
    main()
