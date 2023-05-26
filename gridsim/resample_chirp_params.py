#!/usr/bin/env python

"""
Use the fitted chirp parameters to resample from a real parameter space.
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from eod.communication import chirp_model
from IPython import embed
from rich import print
from scipy.interpolate import interp1d
from scipy.signal.windows import tukey


def load_dataset(path: pathlib.Path):
    files = list(path.glob("*freq_fit.npy"))
    print(f"Found {len(files)} files.")

    params = []
    for file in files:
        param = np.load(file)
        params.append(param)
    params = np.concatenate(params)
    return params


def resample_chirp_params(path: pathlib.Path):
    fits = load_dataset(path)

    params = [
        
    ]

    # remove the nan colums
    fits = fits[~np.isnan(fits).any(axis=1)]

    # sort by chirp duration
    fits = fits[np.argsort(fits[:, 2])]

    # remote outliers in all dimensions
    for i in range(fits.shape[1]):
        fits = fits[
            np.abs(fits[:, i] - np.mean(fits[:, i])) < 10 * np.std(fits[:, i])
        ]

        # specifically remove too high kurt2 values
        if i == 7:
            fits = fits[
                np.abs(fits[:, i] - np.mean(fits[:, i])) < np.std(fits[:, i])
            ]

    # interpolate to make more
    old_x = np.arange(fits.shape[0])
    new_x = np.linspace(0, fits.shape[0], 1000)

    new_fits = []
    for i in range(fits.shape[1]):
        f = interp1d(old_x, fits[:, i], kind="linear", fill_value="extrapolate")
        new_fits.append(f(new_x))
        plt.plot(new_x, new_fits[-1], ".-", label="new")
        plt.plot(old_x, fits[:, i], ".", label="old")
        plt.title(params[i])
        plt.legend()
        plt.show()
    new_fits = np.array(new_fits).T

    print(new_fits.shape)

    # plot the new distributions and overlay the old ones
    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    for i, param in enumerate(params):
        ax[i // 3, i % 3].hist(
            new_fits[:, i], bins=50, density=True, alpha=0.5, label="new"
        )
        ax[i // 3, i % 3].hist(
            fits[:, i], bins=50, alpha=0.5, density=True, label="old"
        )
        ax[i // 3, i % 3].set_title(param)
        ax[i // 3, i % 3].legend()
    plt.tight_layout()
    plt.show()

    # plot the resulting chirps
    fig, ax = plt.subplots()
    t = np.linspace(0, 0.5, 20000) - 0.25
    tuk = tukey(len(t), alpha=0.4)
    for i in range(len(new_fits)):
        popt = new_fits[i]
        chirp = chirp_model(t, *popt)
        mode = np.histogram(chirp, bins=100)[1][
            np.argmax(np.histogram(chirp, bins=100)[0])
        ]
        chirp -= mode
        chirp *= tuk
        ax.plot(t, chirp, alpha=0.1, color="k")
    plt.show()

    np.save(path / "interpolation.npy", new_fits)


def interface():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", "-p", type=pathlib.Path, help="Path to the dataset."
    )
    args = parser.parse_args()
    return args


def main():
    args = interface()
    resample_chirp_params(args.path)


if __name__ == "__main__":
    main()
