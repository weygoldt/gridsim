#!/usr/bin/env python3

"""
Take a simulated dataset and add realistic background noise to it.
Add realistic background noise to a simulated dataset and scale it to
match a real recording by combining the simulated dataset with a real
recording.
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from utils.files import Config, SimulatedDataset, WaveTrackerDataset


def hybrid_grid(simulation_path: pathlib.Path, real_path: pathlib.Path):
    sd = SimulatedDataset(simulation_path)
    rd = WaveTrackerDataset(real_path)

    # normalize the simulated dataset
    sd.track_powers = (sd.track_powers - np.mean(sd.track_powers)) / np.std(
        sd.track_powers
    )
    sd.raw = (sd.raw - np.mean(sd.raw)) / np.std(sd.raw)

    # take a random snippet from the real recording
    target_shape = sd.raw.shape
    source_shape = rd.raw.shape
    random_electrodes = np.random.choice(
        np.arange(source_shape[1]), size=target_shape[1], replace=False
    )
    random_start = np.random.randint(
        0, source_shape[0] - target_shape[0], size=1
    )[0]
    random_end = random_start + target_shape[0]
    real_snippet = rd.raw[random_start:random_end, random_electrodes]

    # get mean and std of the real recording
    mean_power, std_power = np.nanmean(rd.track_powers), np.nanstd(
        rd.track_powers
    )
    mean_amp, std_amp = np.mean(real_snippet), np.std(real_snippet)

    # scale the simulated dataset to match the real recording
    sd.track_powers = sd.track_powers * std_power + mean_power
    sd.raw = sd.raw * std_amp + mean_amp

    # combine the simulated dataset with the real recording
    sd.raw = sd.raw + real_snippet

    # save the hybrid dataset
    sd.save()


def interface():
    parser = argparse.ArgumentParser(
        description="Add realistic background noise to a simulated dataset."
    )
    parser.add_argument(
        "--simulation_path",
        "-s",
        type=pathlib.Path,
        help="Path to the simulated dataset.",
    )
    parser.add_argument(
        "--real_path",
        "-r",
        type=pathlib.Path,
        help="Path to the real recording.",
    )
    args = parser.parse_args()
    return args


def main():
    args = interface()
    hybrid_grid(args.simulation_path, args.real_path)


if __name__ == "__main__":
    main()
