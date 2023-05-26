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
    freq_fits = list(path.glob("*freq_fit.npy"))
    env_fits = list(path.glob("*env_fit.npy"))

    print(f"Found {len(env_fits)} files.")

    freq_params = []
    env_params = []
    for freq, env in zip(freq_fits, env_fits):
        fps = np.load(freq)
        eps = np.load(env)
        freq_params.append(fps)
        env_params.append(eps)
    freq_params = np.concatenate(freq_params)
    env_params = np.concatenate(env_params)
    return freq_params, env_params


def resample_chirp_params(path: pathlib.Path):
    freq_fits, env_fits = load_dataset(path)

    params = [
        "amp1",
        "std1",
        "mu1",
        "kurt1",
        "amp2",
        "std2",
        "mu2",
        "kurt2",
        "amp3",
        "std3",
        "mu3",
        "kurt3",
    ]

    # remove the nan colums in both parameter matrices
    freq_fits = freq_fits[:, ~np.isnan(freq_fits).any(axis=0)]
    env_fits = env_fits[:, ~np.isnan(env_fits).any(axis=0)]

    # sort by std of first gaussian of freq
    env_fits = env_fits[np.argsort(freq_fits[:, 1])]
    freq_fits = freq_fits[np.argsort(freq_fits[:, 1])]

    # remote outliers in all dimensions
    # for i in range(fits.shape[1]):
    #     fits = fits[np.abs(fits[:, i] - np.mean(fits[:, i])) < 10 * np.std(fits[:, i])]

    #     # specifically remove too high kurt2 values
    #     if i == 7:
    #         fits = fits[np.abs(fits[:, i] - np.mean(fits[:, i])) < np.std(fits[:, i])]

    # interpolate to make more
    old_x = np.arange(freq_fits.shape[0])
    new_x = np.linspace(0, freq_fits.shape[0], 1000)

    new_freq_fits = []
    new_env_fits = []

    for i in range(freq_fits.shape[1]):
        ff = interp1d(old_x, freq_fits[:, i], kind="linear", fill_value="extrapolate")
        ef = interp1d(old_x, env_fits[:, i], kind="linear", fill_value="extrapolate")

        new_freq_fits.append(ff(new_x))
        new_env_fits.append(ef(new_x))

        plt.plot(new_x, ff(new_x), label="freq")
        plt.plot(new_x, ef(new_x), label="env")

        plt.title(params[i])
        plt.legend()
        plt.show()

    new_freq_fits = np.array(new_freq_fits).T
    new_env_fits = np.array(new_env_fits).T

    # plot the new distributions and overlay the old ones
    fig, ax = plt.subplots(4, 4, figsize=(20, 20))
    for i, param in enumerate(params):
        ax[i // 4, i % 4].hist(freq_fits[:, i], bins=100, alpha=0.5, label="freq")
        ax[i // 4, i % 4].hist(new_freq_fits[:, i], bins=100, alpha=0.5, label="new freq")
        ax[i // 4, i % 4].set_title(param)
    fig.suptitle("Frequency fits")
    plt.tight_layout()
    plt.show()

    # repeat for env
    fig, ax = plt.subplots(4, 4, figsize=(20, 20))
    for i in range(len(params)):
        ax[i // 4, i % 4].hist(env_fits[:, i], bins=100, alpha=0.5, label="env")
        ax[i // 4, i % 4].hist(new_env_fits[:, i], bins=100, alpha=0.5, label="new env")
        ax[i // 4, i % 4].set_title(params[i])
    fig.suptitle("Envelope fits")
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
    parser.add_argument("--path", "-p", type=pathlib.Path, help="Path to the dataset.")
    args = parser.parse_args()
    return args


def main():
    args = interface()
    resample_chirp_params(args.path)


if __name__ == "__main__":
    main()
