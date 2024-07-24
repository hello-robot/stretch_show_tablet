import numpy as np

JOINT_NAME_SHORT_TO_FULL = {
    "base": "rotate_mobile_base",
    "lift": "joint_lift",
    "arm_extension": "wrist_extension",
    "yaw": "joint_wrist_yaw",
    "pitch": "joint_wrist_pitch",
    "roll": "joint_wrist_roll",
}

# define joint limits from above function as a dict
JOINT_LIMITS = {
    "base": (-np.pi, np.pi),
    "lift": (0.25, 1.1),
    "arm_extension": (0.02, 0.45),
    "yaw": (-np.deg2rad(60.0), np.pi),
    "pitch": (-np.pi / 2, 0.2),
    "roll": (-np.pi/2, np.pi/2),
}
