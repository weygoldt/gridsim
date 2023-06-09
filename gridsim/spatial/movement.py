from dataclasses import dataclass, field
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from scipy.interpolate import interp1d
from scipy.stats import gamma, norm

np.random.seed(42)


@dataclass
class MovementParams:
    """
    All parameters for simulating the movement of a fish.
    """

    duration: int = 600
    origin: Tuple[float, float] = (0, 0)
    boundaries: Tuple[float, float, float, float] = (-5, 5, -5, 5)
    forward_s: float = 0.2
    backward_s: float = 0.1
    backward_h: float = 0.1
    mode_veloc: float = 0.2
    max_veloc: float = 1
    measurement_fs: float = 30
    target_fs: int = 30


def directionPDF(
    forward_s,
    backward_s,
    backward_h,
    measurement_fs=30,
    target_fs=3,
):
    forward_s = np.sqrt((1 / target_fs) / (1 / measurement_fs)) * forward_s
    backward_s = np.sqrt((1 / target_fs) / (1 / measurement_fs)) * backward_s

    directions = np.arange(0, 2 * np.pi, 0.0001)
    p_forward1 = norm.pdf(directions, 0, forward_s)
    p_forward2 = norm.pdf(directions, np.max(directions), forward_s)
    p_backward = norm.pdf(directions, np.pi, backward_s) * backward_h
    probabilities = (p_forward1 + p_forward2 + p_backward) / np.sum(
        p_forward1 + p_forward2 + p_backward
    )

    return directions, probabilities


def stepPDF(max_veloc: float, duration: int, target_fs: int = 3) -> np.ndarray:
    """
    Generate a sequence of steps representing the steps of a random walker.
    Step lengths are drawn from a gamma distribution.

    Parameters
    ----------
    peak_veloc : float
        Peak velocity of the step function (mode of the gamma distribution).
    max_veloc : float
        Maximum velocity of the step function (maximum value of the gamma distribution).
    duration : int
        Duration of the step function in seconds.
    target_fs : int, optional
        Sampling frequency of the step function (default is 3 Hz).

    Returns
    -------
    np.ndarray
        Array of step values.

    """

    # this is just a random gamma distribution
    # in the future, fit one to the real data and use that one
    g = gamma.rvs(a=5, scale=1, size=(duration * target_fs) - 1)

    # scale the gamma distribution to the desired peak velocity
    g = g * (max_veloc / np.max(g))

    # scale the meters per second to the time step specified by the
    # target sampling frequency
    g = g * (1 / target_fs)

    return g


def steps(MovementParams):
    """
    Makes steps (distance) and directions (trajectories) of a random walker
    in a 2D space.

    Parameters
    ----------
    duration : int, optional
        Total duration of simulation, by default 600
    fs : int, optional
        Sampling rate, by default 30
    species : str, optional
        The species key, by default "Alepto"
    plot : bool, optional
        Enable or disable plotting the probability distributions
        underlying the parameters, by default False

    Returns
    -------
    Tuple(np.ndarray, np.ndarray)
        Tuple of trajectories and steps.

    """
    # get the probability distribution of directions
    directions, probabilities = directionPDF(
        MovementParams.forward_s,
        MovementParams.backward_s,
        MovementParams.backward_h,
        MovementParams.measurement_fs,
        MovementParams.measurement_fs,
    )

    # make random step lengths according to a gamma distribution
    steps = stepPDF(
        MovementParams.max_veloc,
        MovementParams.duration,
        MovementParams.measurement_fs,
    )

    # draw random directions according to the probability distribution
    trajectories = np.random.choice(
        directions,
        size=(MovementParams.duration * MovementParams.measurement_fs) - 1,
        p=probabilities,
    )

    return trajectories, steps


def positions(
    trajectories: np.ndarray,
    steps: np.ndarray,
    MovementParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates a random walk with a given set of trajectories and step sizes.
    Given an origin position, boundaries, a set of trajectories, and a set of
    step sizes, this function computes the final x and y positions of the
    trajectories after taking steps and folding back to the boundaries.

    Parameters
    ----------
    origin : Tuple[float, float]
        The (x, y) starting position of the agent.

    boundaries : Tuple[float, float, float, float]
        The minimum and maximum x and y positions allowed for the positions.
        Everything outside these boundaries will be folded back to the boundaries.

    trajectories : np.ndarray
        A 1D array of angle values in radians specifying the direction of each
        step in the trajectory.

    steps : np.ndarray
        A 1D array of step sizes for each step in the trajectory.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of two 1D arrays representing the final x and y
        positions of the trajectories.
    """
    origin = MovementParams.origin
    boundaries = MovementParams.boundaries

    x = np.full(len(trajectories) + 1, np.nan)
    y = np.full(len(trajectories) + 1, np.nan)
    x[0] = origin[0]
    y[0] = origin[1]

    for i in range(len(trajectories)):
        # use the first trajectory as is
        if i == 0:
            converted_trajectory = trajectories[i]

        # make all other trajectories relative to the previous one
        else:
            converted_trajectory = trajectories[i - 1] - trajectories[i]

            # make sure the trajectory is between 0 and 2pi
            if converted_trajectory > 2 * np.pi:
                converted_trajectory = converted_trajectory - 2 * np.pi
            if converted_trajectory < 0:
                converted_trajectory = converted_trajectory + 2 * np.pi

            # write current trajectory to trajectories to correct
            # future trajectories relative to the current one
            trajectories[i] = converted_trajectory

        # use trigonometric identities to calculate the x and y positions
        y[i + 1] = np.sin(converted_trajectory) * steps[i]
        x[i + 1] = np.cos(converted_trajectory) * steps[i]

    # cumulatively add the steps to the positions
    x = np.cumsum(x)
    y = np.cumsum(y)

    # interpolate the position to the target fs
    time_sim = np.arange(
        0, MovementParams.duration, 1 / MovementParams.measurement_fs
    )
    time_target = np.arange(
        0, MovementParams.duration, 1 / MovementParams.target_fs
    )
    xinterper = interp1d(time_sim, x, kind="cubic", fill_value="extrapolate")
    yinterper = interp1d(time_sim, y, kind="cubic", fill_value="extrapolate")

    x = xinterper(time_target)
    y = yinterper(time_target)

    # fold back the positions if they are outside the boundaries
    boundaries = np.ravel(boundaries)
    while (
        np.any(x < boundaries[0])
        or np.any(x > boundaries[1])
        or np.any(y < boundaries[2])
        or np.any(y > boundaries[3])
    ):
        x[x < boundaries[0]] = boundaries[0] + (
            boundaries[0] - x[x < boundaries[0]]
        )
        x[x > boundaries[1]] = boundaries[1] - (
            x[x > boundaries[1]] - boundaries[1]
        )
        y[y < boundaries[2]] = boundaries[2] + (
            boundaries[2] - y[y < boundaries[2]]
        )
        y[y > boundaries[3]] = boundaries[3] - (
            y[y > boundaries[3]] - boundaries[3]
        )

    return x, y


def main():
    n_fish = 3
    fig, ax = plt.subplots()

    for _ in range(n_fish):
        trajectories, s = steps()
        x, y = positions(trajectories, s)
        ax.scatter(x, y, marker=".")

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    main()
