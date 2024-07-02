import numpy as np

from human import Human
from utils import Direction, in_range, get_vector_direction_image_plane
    
class TabletController:
    def __init__(self):
        self.tablet_horizontal_deadband = [-0.1, 0.1]
        self.tablet_portrait_x_offset = 0.15

    def get_head_vertical_vector(self, human: Human):
        face_landmarks = human.pose_estimate.face_estimate
        chin_xyz = np.array(face_landmarks["chin_middle"])
        nose_xyz = np.array(face_landmarks["nose_tip"])
        return nose_xyz - chin_xyz

    def get_head_direction(self, human: Human):
        face_vector = self.get_head_vertical_vector(human)
        return get_vector_direction_image_plane(face_vector)

    def get_tablet_yaw_action(self, human: Human):
        """
        Returns rad
        """
        face_landmarks = human.pose_estimate.face_estimate
        chin_xyz = face_landmarks["chin_middle"]

        head_direction = self.get_head_direction(human)

        if head_direction == Direction.UP:
            x = chin_xyz[0]
        elif head_direction == Direction.LEFT:
            x = chin_xyz[1] + self.tablet_portrait_x_offset
        elif head_direction == Direction.RIGHT:
            x = -chin_xyz[1] - self.tablet_portrait_x_offset
        else:
            return 0.

        if in_range(x, self.tablet_horizontal_deadband):
            return 0.
        
        Kp = 0.2
        yaw_action = Kp * (-1 * x)
        return yaw_action