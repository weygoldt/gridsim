#!/usr/bin/env python3

"""
The base class for fish objects.
"""

from dataclasses import dataclass

import numpy as np
from eod.communication import chirps, rises
from eod.eod import eod
from scipy.interpolate import interp1d
from spatial.movement import positions, steps
from utils.files import Config

conf = Config("config.yml")


class Fish:
    def __init__(
        self,
        fish,
        eodf,
        ChirpParams,
        RiseParams,
        MovementParams,
    ) -> None:
        self.name = getattr(conf.species, fish)
        self.fish_params = getattr(conf.fish, fish)
        self.chirps_params = ChirpParams
        self.rises_params = RiseParams
        self.movement_params = MovementParams
        self.eodf = eodf

        amplitudes = self.fish_params.eod.harmonics.amplitudes
        phases = self.fish_params.eod.harmonics.phases
        self.chirp_trace, self.chirp_amp = chirps(self.chirps_params)
        self.rise_trace = rises(self.rises_params)

        self.frequency = self.chirp_trace + self.rise_trace + eodf
        self.frequency_track = self.rise_trace + eodf

        signal = eod(
            amplitudes=amplitudes,
            phases=phases,
            frequency=self.frequency,
            samplerate=self.chirps_params.samplerate,
            duration=self.chirps_params.duration,
            phase0=self.fish_params.eod.phase0,
            noise_std=self.fish_params.eod.noise_std,
        )

        self.eod = signal * self.chirp_amp

        t, s = steps(MovementParams)
        x, y = positions(t, s, MovementParams)

        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"FishSignal({self.name})"

    def __str__(self) -> str:
        return f"FishSignal({self.name})"
