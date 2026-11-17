from model import LinearRegression
import math
import sys


def main() -> None:
    model = LinearRegression()
    model.load()
    try:
        print("Insert kilometer:")
        kilometer = float(input())
        if (kilometer < 0 or not math.isfinite(kilometer)):
            raise ValueError("Invalid Mileage")
    except (ValueError, OSError, EOFError, KeyboardInterrupt, TypeError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    result = model.hypothese(kilometer)
    print(result)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Ctrl C detected, end of program")
        sys.exit(1)
