#!/usr/bin/env python3

"""
Generate a random grid recording based on the parameters from the 
configuration file.
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
from eod.communication import ChirpParams, RiseParams
from eod.fish import Fish
from IPython import embed
from rich.console import Console
from rich.progress import track
from scipy.signal import resample
from spatial.grid import grid
from spatial.movement import MovementParams
from utils.files import Config
from utils.filters import lowpass_filter

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
        chirp_sizes=csizes,
        chirp_widths=cdurations,
        chirp_kurtosis=ckurtosis,
        chirp_contrasts=ccontrasts,
        chirp_undershoots=cundershoots,
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
        rise_sizes=rsizes,
        rise_taus=rrise_taus,
        decay_taus=rdecay_taus,
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
    nfish = np.random.randint(
        conf.randgrid.fish.nfish[0],
        conf.randgrid.fish.nfish[1],
    )

    ex, ey = grid(
        conf.randgrid.space.origin,
        conf.randgrid.space.grid,
        conf.randgrid.space.spacing,
        conf.randgrid.space.shape,
    )

    track_freqs = []
    track_powers = []
    ypos = []
    xpos = []
    track_idents = []
    track_indices = []
    chirp_times = []
    chirp_sizes = []
    chirp_undershoots = []
    chirp_widths = []
    chirp_ids = []
    rise_times = []
    rise_ids = []

    nelectrodes = len(np.ravel(ex))
    for iter in track(range(nfish), description=f"Generating {nfish} fish"):
        # generate a random fish
        fish = randfish(conf)

        # compute the distance at every position to every electrode
        dists = np.sqrt(
            (fish.x[:, None] - ex[None, :]) ** 2 + (fish.y[:, None] - ey[None, :]) ** 2
        )

        # make the distance sqared and invert it
        dists = -(dists**2)

        # truncate at -1 and shift up to make it multipliable
        dists[dists < -1] = -1
        dists = dists + 1

        # add the fish signal onto all electrodes
        grid_signals = np.tile(fish.eod, (nelectrodes, 1)).T

        # attentuate the signals by the squared distances
        attenuated_signals = grid_signals * dists

        # collect signals
        if iter == 0:
            signal = attenuated_signals
        else:
            signal += attenuated_signals

        # downsample the tracking arrays
        num = int(
            np.round(
                conf.randgrid.time.track_samplerate
                / conf.randgrid.time.samplerate
                * len(fish.frequency)
            )
        )
        f = resample(fish.frequency_track, num)
        p = resample(dists, num, axis=0)
        x = resample(fish.x, num)
        y = resample(fish.y, num)

        # filter to remove resampling artifacts
        f = lowpass_filter(f, 10, conf.randgrid.time.track_samplerate)
        f[f < fish.eodf] = fish.eodf
        p = np.vstack(
            [lowpass_filter(pi, 10, conf.randgrid.time.track_samplerate) for pi in p.T]
        ).T

        p[p < 0] = 0

        track_freqs.append(f)
        track_powers.append(p)
        xpos.append(x)
        ypos.append(y)
        track_idents.append(np.ones_like(f) * iter)
        track_indices.append(np.arange(len(f)))
        chirp_times.append(fish.chirps_params.chirp_times)
        chirp_widths.append(fish.chirps_params.chirp_widths)
        chirp_sizes.append(fish.chirps_params.chirp_sizes)
        chirp_undershoots.append(fish.chirps_params.chirp_undershoots)
        chirp_ids.append(np.ones_like(fish.chirps_params.chirp_times) * iter)
        rise_times.append(fish.rises_params.rise_times)
        rise_ids.append(np.ones_like(fish.rises_params.rise_times) * iter)

    track_freqs = np.concatenate(track_freqs)
    track_powers = np.concatenate(track_powers)
    xpos = np.concatenate(xpos)
    ypos = np.concatenate(ypos)
    track_idents = np.concatenate(track_idents)
    track_indices = np.concatenate(track_indices)
    chirp_times = np.concatenate(chirp_times)
    chirp_widths = np.concatenate(chirp_widths)
    chirp_heights = np.concatenate(chirp_sizes)
    chirp_undershoots = np.concatenate(chirp_undershoots)
    chirp_ids = np.concatenate(chirp_ids)
    rise_times = np.concatenate(rise_times)
    rise_ids = np.concatenate(rise_ids)
    times = np.arange(len(signal)) / conf.randgrid.time.samplerate
    t = np.arange(0, times[-1], 1 / conf.randgrid.time.track_samplerate)

    outpath = pathlib.Path(conf.randgrid.outpath)
    outpath.mkdir(parents=True, exist_ok=True)
    np.save(outpath / "raw.npy", signal)
    np.save(outpath / "times.npy", t)
    np.save(outpath / "fund_v.npy", track_freqs)
    np.save(outpath / "sign_v.npy", track_powers)
    np.save(outpath / "xpos.npy", xpos)
    np.save(outpath / "ypos.npy", ypos)
    np.save(outpath / "ident_v.npy", track_idents)
    np.save(outpath / "idx_v.npy", track_indices)
    np.save(outpath / "chirp_times_gt.npy", chirp_times)
    np.save(outpath / "chirp_widths_gt.npy", chirp_widths)
    np.save(outpath / "chirp_heights_gt.npy", chirp_heights)
    np.save(outpath / "chirp_undershoots_gt.npy", chirp_undershoots)
    np.save(outpath / "chirp_ids_gt.npy", chirp_ids)
    np.save(outpath / "rise_times_gt.npy", rise_times)
    np.save(outpath / "rise_ids_gt.npy", rise_ids)

    con.print(f"Saved {nfish} fish")


def main():
    rand_grid()


if __name__ == "__main__":
    main()
