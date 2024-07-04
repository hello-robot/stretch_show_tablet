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