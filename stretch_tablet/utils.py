import numpy as np
from enum import Enum
import json

# enums
class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    IN = 5
    OUT = 6

# i/o
def load_bad_json_data(data_string):
    data_string = data_string.replace("'", "\"")
    data_string = data_string.replace("(", "[")
    data_string = data_string.replace(")", "]")

    data = json.loads(data_string)
    return data

def load_bad_json(file):
    with open(file) as f:
        data_string = json.load(f)
        return load_bad_json_data(data_string)

landmark_names = ['nose', 'neck',
                'right_shoulder', 'right_elbow', 'right_wrist',
                'left_shoulder', 'left_elbow', 'left_wrist',
                'right_hip', 'right_knee', 'right_ankle',
                'left_hip', 'left_knee', 'left_ankle',
                'right_eye', 'left_eye',
                'right_ear', 'left_ear']

# math
def spherical_to_cartesian(radius, azimuth, elevation):
    """
    Convert spherical coordinates to Cartesian coordinates.
    
    Args:
        radius (float): The radius or radial distance.
        azimuth (float): The azimuth angle in radians.
        elevation (float): The elevation angle in radians.
    
    Returns:
        list: A list containing the Cartesian coordinates (x, y, z).
    """
    x = radius * np.sin(elevation) * np.cos(azimuth)
    y = radius * np.sin(elevation) * np.sin(azimuth)
    z = radius * np.cos(elevation)
    
    return [x, y, z]

def in_range(value, range):
    return True if value >= range[0] and value <= range[1] else False

def get_vector_direction_image_plane(v: np.ndarray) -> Direction:
    x = v[0]
    y = v[1]
    
    theta = np.arctan2(y, x)

    PI_4 = np.pi / 4.
    if in_range(theta, [-PI_4, PI_4]):
        return Direction.RIGHT
    elif in_range(theta, [-3*PI_4, -PI_4]):
        return Direction.DOWN
    elif in_range(theta, [PI_4, 3*PI_4]):
        return Direction.UP
    else:
        return Direction.LEFT
