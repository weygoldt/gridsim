import numpy as np


def grid(origin, shape, spacing, type="hex"):
    assert type in ["hex", "square"], "type must be 'hex' or 'square'"
    assert (shape[0] % 2 == 0) or (type == "square"), "shape must be even"

    # grid parameters
    electrode_number = shape[0] * shape[1]
    electrode_index = np.arange(0, electrode_number)
    electrode_x = np.mod(electrode_index, shape[0]) * spacing
    electrode_y = np.floor(electrode_index / shape[0]) * spacing

    # shift every second row to make a hexagonal grid
    if type == "hex":
        electrode_y[1::2] += spacing / 2

    # shift the grid to the specified origin
    electrode_x += origin[0] - np.mean(electrode_x)
    electrode_y += origin[1] - np.mean(electrode_y)

    return electrode_x, electrode_y
