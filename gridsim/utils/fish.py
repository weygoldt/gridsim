#!/usr/bin/env python3

"""
The base class for fish objects.
"""

from dataclasses import dataclass

from eod.communication import ChirpParams, RiseParams, chirps, rises
from eod.eod import eod
from utils.files import Config

conf = Config("config.yml")


def fish_eod(conf, ChirpParams, RiseParams):
    rise_trace = rises(RiseParams)
    chirp_trace, chirp_amp = chirps(ChirpParams)


class Fish:
    def __init__(self, params: dict) -> None:
        pass
