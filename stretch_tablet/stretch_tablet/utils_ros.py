import numpy as np
import sophuspy as sp
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import PoseStamped, Point, Quaternion

def point2tuple(p: Point):
    return [p.x, p.y, p.z]

def quat2tuple(q: Quaternion):
    return [q.x, q.y, q.z, q.w]

def generate_pose_stamped(position, orientation, timestamp):
    pose_stamped = PoseStamped()
    point = Point()
    point.x = position[0]
    point.y = position[1]
    point.z = position[2]
    quat = Quaternion()
    quat.x = orientation[0]
    quat.y = orientation[1]
    quat.z = orientation[2]
    quat.w = orientation[3]
    pose_stamped.pose.position = point
    pose_stamped.pose.orientation = quat
    pose_stamped.header.stamp = timestamp
    return pose_stamped

# conversions
def posestamped2se3(pose_stamped: PoseStamped) -> sp.SE3:
    pos = pose_stamped.pose.position
    ori = pose_stamped.pose.orientation
    pos_np = np.array(point2tuple(pos))
    ori_np = np.array(quat2tuple(ori))
    ori_R = R.from_quat(ori_np).as_matrix()
    return sp.SE3(ori_R, pos_np)
