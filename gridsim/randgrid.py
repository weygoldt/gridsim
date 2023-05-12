#!/usr/bin/env python3

"""
Generate a random grid recording based on the parameters from the 
configuration file.
"""

import numpy as np
from eod.communication import ChirpParams, RiseParams
from fish import Fish
from IPython import embed
from rich.console import Console
from rich.progress import track
from spatial.grid import grid
from spatial.movement import MovementParams
from utils.files import Config

conf = Config("config.yml")
con = Console()

np.random.seed(conf.meta.random_seed)


def rand_chirps(conf: Config) -> ChirpParams:
    duration = conf.randgrid.time.duration
    samplerate = conf.randgrid.time.samplerate
    fish = getattr(conf.fish, conf.randgrid.fish.species)

    nchirps = np.random.randint(
        conf.randgrid.fish.nchirps[0],
        conf.randgrid.fish.nchirps[1],
    )
    ctimes = np.random.uniform(
        0.0,
        duration,
        size=nchirps,
    )
    csizes = np.random.uniform(
        fish.eod.chirp.sizes[0],
        fish.eod.chirp.sizes[1],
        size=nchirps,
    )
    cdurations = np.random.uniform(
        fish.eod.chirp.durations[0],
        fish.eod.chirp.durations[1],
        size=nchirps,
    )
    ckurtosis = np.random.uniform(
        fish.eod.chirp.kurtosis[0],
        fish.eod.chirp.kurtosis[1],
        size=nchirps,
    )
    ccontrasts = np.random.uniform(
        fish.eod.chirp.contrasts[0],
        fish.eod.chirp.contrasts[1],
        size=nchirps,
    )
    cundershoots = np.random.uniform(
        fish.eod.chirp.undershoots[0],
        fish.eod.chirp.undershoots[1],
        size=nchirps,
    )
    cp = ChirpParams(
        eodf=0.0,
        samplerate=samplerate,
        duration=duration,
        chirp_times=ctimes,
        chirp_size=csizes,
        chirp_width=cdurations,
        chirp_kurtosis=ckurtosis,
        chirp_contrasts=ccontrasts,
        chirp_undershoot=cundershoots,
    )

    return cp


def rand_rises(conf: Config) -> RiseParams:
    duration = conf.randgrid.time.duration
    samplerate = conf.randgrid.time.samplerate
    fish = getattr(conf.fish, conf.randgrid.fish.species)

    nrises = np.random.randint(
        conf.randgrid.fish.nrises[0],
        conf.randgrid.fish.nrises[1],
    )
    rtimes = np.random.uniform(
        0.0,
        duration,
        size=nrises,
    )
    rsizes = np.random.uniform(
        fish.eod.rise.sizes[0],
        fish.eod.rise.sizes[1],
        size=nrises,
    )
    rrise_taus = np.random.uniform(
        fish.eod.rise.rise_tau[0],
        fish.eod.rise.rise_tau[1],
        size=nrises,
    )
    rdecay_taus = np.random.uniform(
        fish.eod.rise.decay_tau[0],
        fish.eod.rise.decay_tau[1],
        size=nrises,
    )
    rp = RiseParams(
        eodf=0.0,
        samplerate=samplerate,
        duration=duration,
        rise_times=rtimes,
        rise_size=rsizes,
        rise_tau=rrise_taus,
        decay_tau=rdecay_taus,
    )

    return rp


def randfish(conf):
    fishparams = getattr(conf.fish, conf.randgrid.fish.species)
    cps = rand_chirps(conf)
    rps = rand_rises(conf)

    origin_x = np.random.uniform(
        conf.randgrid.space.limits[0],
        conf.randgrid.space.limits[1],
    )
    origin_y = np.random.uniform(
        conf.randgrid.space.limits[0],
        conf.randgrid.space.limits[1],
    )

    mvm = MovementParams(
        duration=conf.randgrid.time.duration,
        origin=(origin_x, origin_y),
        boundaries=conf.randgrid.space.limits,
        forward_s=fishparams.movement.forward_s,
        backward_s=fishparams.movement.backward_s,
        backward_h=fishparams.movement.backward_h,
        mode_veloc=fishparams.movement.mode_veloc,
        max_veloc=fishparams.movement.max_veloc,
        measurement_fs=20,
        target_fs=conf.randgrid.time.samplerate,
    )

    eodf = np.random.uniform(
        fishparams.eod.frequency[0],
        fishparams.eod.frequency[1],
    )

    fish = Fish(
        fish=conf.randgrid.fish.species,
        eodf=eodf,
        ChirpParams=cps,
        RiseParams=rps,
        MovementParams=mvm,
    )

    return fish


def rand_grid():
    import matplotlib.pyplot as plt

    nfish = np.random.randint(
        conf.randgrid.fish.nfish[0],
        conf.randgrid.fish.nfish[1],
    )

    electrode_x, electrode_y = grid(
        conf.randgrid.space.origin,
        conf.randgrid.space.grid,
        conf.randgrid.space.spacing,
        conf.randgrid.space.shape,
    )

    fig, ax = plt.subplots()
    for iter in track(range(nfish), description="Generating fish"):
        fish = randfish(conf)
        ax.plot(fish.x, fish.y, alpha=0.5)

    ax.scatter(electrode_x, electrode_y, marker="x", color="red", zorder=10)
    ax.set_aspect("equal")
    plt.show()

    con.log(f"Generated {nfish} fish")


def main():
    rand_grid()
    pass


if __name__ == "__main__":
    main()
