"""Setup and run exercise examples"""

# pylint: disable=unused-variable, invalid-name, too-many-locals
import argparse  # noqa: E402
import os
import pickle  # noqa: E402
from urllib.request import urlretrieve  # noqa: E402

import numpy as np  # noqa: E402
import rnnutils  # noqa: E402
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.models import Sequential


def download_airline_passengers(fn, site):
    """Download airline passengers file"""
    url = os.path.join(site, fn)
    urlretrieve(url, "airline-passengers.csv")


def airlines():  # noqa: R0914
    """Run airlines exercise"""
    rnnutils.plt.figure(figsize=(10, 6))
    with open("airline-model.pkl", "rb") as f:
        pdata = pickle.load(f)
    fn = "airline-passengers.csv"
    if not os.path.exists(fn):
        site = "https://raw.githubusercontent.com/jbrownlee/Datasets/master"
        download_airline_passengers(fn, site)

    # Plot airline data
    df = rnnutils.airlines()
    _, ax = rnnutils.plt.subplots(figsize=(10, 6))
    ax.plot(df.time, df.passengers)
    rnnutils.plt.tick_params(axis="both", which="major", labelsize=20)
    rnnutils.plt.savefig("airline.png")

    # Plot example of train and test data
    fig, ax = rnnutils.plt.subplots(figsize=(10, 6))
    ax.plot(df.time, df.passengers)
    ax.plot(df.time[100:144], df.passengers[100:144], color="red")
    ax.legend(["train", "test"], fontsize=20)
    rnnutils.plt.tick_params(axis="both", which="major", labelsize=20)
    rnnutils.plt.savefig("airline-train-test.png")

    # Prepare train and test data
    data = np.array(df["passengers"].values.astype("float32")).reshape(-1, 1)
    train, test, scaler = rnnutils.make_train_test(data)

    # Make xy training test data
    time_steps = 12
    trainX, trainY, trainX_indices, trainY_indices = rnnutils.make_xy(
        train, time_steps
    )  # noqa W0612, C0103
    testX, testY, testX_indices, testY_indices = rnnutils.make_xy(
        test, time_steps
    )  # noqa W0612, C0103

    # Define model
    model = Sequential()
    model.add(SimpleRNN(units=3, input_shape=(time_steps, 1), activation="tanh"))
    model.add(Dense(units=1, activation="tanh"))
    model.compile(loss="mean_squared_error", optimizer="adam")

    # Fit model
    history = model.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)
    # Plot model training history
    rnnutils.plot_history(history, show=False)
    rnnutils.plt.savefig("airline-training-history.png")

    # Plot model prediction
    data = {
        "train": (model.predict(trainX), train, trainY_indices),
        "test": (model.predict(testX), test, testY_indices),
    }
    rnnutils.plot_pred(
        data,
        scaler=scaler,
        ticks=range(0, 144, 20),
        labels=df.year[range(0, 144, 20)],
        show=False,
    )
    rnnutils.plt.savefig("airline-prediction.png")

    # Plot model fit, prepared data
    rnnutils.plot_pred(
        {"train": data["train"], "test": data["test"]},
        scaler=scaler,
        ticks=range(0, 144, 20),
        labels=df.year[range(0, 144, 20)],
        show=False,
    )
    rnnutils.plt.savefig("airline-prediction-prepared.png")

    # Plot model history, loss, prepared data
    rnnutils.plot_history({"loss": pdata["loss"]}, show=False)
    rnnutils.plt.savefig("airline-history-prepared.png")


def alphabet():
    """Run alphabet exercise"""


def main():
    """Setup and run airline example"""
    parser = argparse.ArgumentParser(
        prog="airline.py",
        description="Setup and run airline example",
    )
    parser.add_argument("model", choices=["both", "airline", "alphabet"])
    args = parser.parse_args()

    if args.model == "both":
        airlines()
        alphabet()
    elif args.model == "airline":
        airlines()
    elif args.model == "alphabet":
        alphabet()
    else:
        print("Unknown model")


if __name__ == "__main__":
    main()
