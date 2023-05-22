#!/usr/bin

"""
This file extracts the evolution of the instantaneous frequency and amplitude
of chirps from a real labeled dataset. The extracted parameters can then 
be used to simulate chirps with the same characteristics.
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from rich import print

from gridtools.utils.datasets import WTRaw
from gridtools.utils.filters import bandpass_filter


def get_upper_fish(dataset):
    min_fs = []
    track_ids = np.unique(dataset.track_idents[~np.isnan(dataset.track_idents)])
    for track_id in track_ids:
        f = dataset.track_freqs[dataset.track_idents == track_id]
        min_fs.append(np.min(f))
    return track_ids[np.argmax(min_fs)]


def extract_features(dataset):
    time_window = 1.0

    upper_fish = get_upper_fish(dataset)
    chirp_times = dataset.chirp_times_cnn[dataset.chirp_idents_cnn == upper_fish]
    track_freqs = dataset.track_freqs[dataset.track_idents == upper_fish]
    track_times = dataset.track_times[
        dataset.track_indices[dataset.track_idents == upper_fish]
    ]
    track_powers = dataset.track_powers[
        dataset.track_indices[dataset.track_idents == upper_fish, :]
    ]

    for chirp in chirp_times:
        track_index = np.argmin(np.abs(track_times - chirp))
        track_freq = track_freqs[track_index]
        track_power = track_powers[track_index, :]
        best_electrode = np.argmax(track_power)

        raw_index = np.arange(
            chirp - time_window / 2, chirp + time_window / 2, 1 / dataset.samplerate
        )
        raw = dataset.raw[raw_index, best_electrode]
        raw = bandpass_filter(raw, track_freq - 50, track_freq + 300, dataset.samplerate)


def interface():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=pathlib.Path, help="Path to dataset.")
    parser.add_argument("--output", "-o", type=pathlib.Path, help="Path to output.")
    args = parser.parse_args()
    return args


def main():
    args = interface()
    dataset = WTRaw(args.input)
    features = extract_features(dataset)
    np.save(args.output, features)


if __name__ == "__main__":
    main()
