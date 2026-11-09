"""The existing Atlas loader's two random draws, with canonical cache keys."""

import torch

PLANES = ((1, 2), (1, 3), (2, 3))
ORIENTATIONS = ((0, 0),) + tuple((plane, turns) for plane in range(3) for turns in range(1, 4))


def orientation_key(plane, turns):
    if plane not in range(3) or turns not in range(4):
        raise ValueError("Invalid Atlas rotation")
    return "identity" if turns == 0 else f"plane{plane}_turns{turns}"


def draw_orientations(count, generator, training=True):
    # Keep both draws, their order, and their original ranges. In particular,
    # identity retains probability 3/12; do not draw uniformly from ten keys.
    result = []
    for _ in range(count):
        if training and generator is not None:
            plane = int(torch.randint(0, 3, (1,), generator=generator).item())
            turns = int(torch.randint(0, 4, (1,), generator=generator).item())
        else:
            plane, turns = 0, 0
        result.append(orientation_key(plane, turns))
    return result


def rotate_volume(volume, plane, turns):
    orientation_key(plane, turns)
    return torch.rot90(volume, turns, PLANES[plane]).contiguous()


def augmentation_contract():
    return {"planes_cdhw": [list(p) for p in PLANES], "turns": [0, 1, 2, 3],
            "draw_order": ["uniform_plane_0_2", "uniform_turns_0_3"],
            "orientations": [orientation_key(*item) for item in ORIENTATIONS],
            "identity_probability": "3/12", "other_orientation_probability": "1/12",
            "validation_orientation": "identity"}
