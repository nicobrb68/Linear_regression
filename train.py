from model import LinearRegression
import sys


def main() -> None:
    model = LinearRegression()
    model.load_dataset("data.csv")
    model.train()
    model.save()
    score = model.precision()
    print(f"Model Precision (R2) : {score * 100:2f}%")
    model.plot()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Ctrl C detected, end of program :")
        sys.exit(1)
