import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import sophuspy as sp
import numpy as np
from scipy.spatial.transform import Rotation as R

import json
from stretch_tablet.human import Human
from stretch_tablet.planner import TabletPlanner
from stretch_tablet.utils import load_bad_json_data

from stretch_tablet_interfaces.srv import PlanTabletPose

class ShowTabletNode(Node):
    def __init__(self):
        super().__init__("show_tablet_node")

        # pub
        self.pub_tablet_goal = self.create_publisher(
            String,
            "/stretch_tablet/goal",
            qos_profile=1
        )

        # sub
        self.sub_face_landmarks = self.create_subscription(
            String,
            "/faces/landmarks_3d",
            callback=self.callback_face_landmarks,
            qos_profile=1
        )

        self.sub_body_landmarks = self.create_subscription(
            String,
            "/body_landmarks/landmarks_3d",
            callback=self.callback_body_landmarks,
            qos_profile=1
        )

        # srv
        self.srv_plan_tablet_pose = self.create_client(PlanTabletPose, 'plan_tablet_pose')

        # tf
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(buffer=self.tf_buffer, node=self)
        # time.sleep(1.)

        # state
        self.face_landmarks = None
        self.body_landmarks = None
        self.latest_camera_pose = None
        self.planner = TabletPlanner()
        self.human = Human()

    # callbacks
    def callback_face_landmarks(self, msg: String):
        msg_data = msg.data.replace("\"", "")
        if msg_data == "None" or msg_data is None:
            return

        data = load_bad_json_data(msg_data)

        if data is None:
            return
        self.human.pose_estimate.set_face_estimate(data)

    def callback_body_landmarks(self, msg: String):
        msg_data = msg.data.replace("\"", "")
        if msg_data == "None" or msg_data is None:
            return

        data = load_bad_json_data(msg_data)

        if data is None or data == "{}":
            return
        self.human.pose_estimate.set_body_estimate(data)
    
    def clear_landmarks(self):
        self.human.pose_estimate.clear_estimates()

    def go_to_pose(self, pose):
        pass

    def build_tablet_pose_request(self):
        request = PlanTabletPose.Request()
        human = self.human

        # extract request info
        body_string = json.dumps(human.pose_estimate.body_estimate)
        camera_pose = human.pose_estimate.get_camera_pose()
        camera_position = camera_pose.translation()
        camera_orientation = R.from_matrix(camera_pose.rotationMatrix()).as_quat()

        # construct request
        request.human_joint_dict = body_string
        request.camera_position = [v for v in camera_position]
        request.camera_orientation = [v for v in camera_orientation]

        return request

    # main
    def main(self):
        while rclpy.ok():
            try:
                t = self.tf_buffer.lookup_transform(
                        # "camera_color_optical_frame",
                        # "odom",
                        "odom",
                        "camera_color_optical_frame",
                        rclpy.time.Time())
                
                camera_pos = t.transform.translation
                camera_ori = t.transform.rotation
                # self.latest_camera_pose = [[camera_pos.x, camera_pos.y, camera_pos.z],
                #                             [camera_ori.x, camera_ori.y, camera_ori.z, camera_ori.w]]
                camera_position = np.array([camera_pos.x, camera_pos.y, camera_pos.z])
                camera_orientation = R.from_quat([camera_ori.x, camera_ori.y, camera_ori.z, camera_ori.w]).as_matrix()
                world2camera_pose = sp.SE3(camera_orientation, camera_position)
                self.human.pose_estimate.set_camera_pose(world2camera_pose)

                if self.human.pose_estimate.is_populated():
                    plan_request = self.build_tablet_pose_request()
                    future = self.srv_plan_tablet_pose.call_async(plan_request)

                    # wait for srv to finish
                    while rclpy.ok():
                        rclpy.spin_once(self)
                        if future.done():
                            try:
                                response = future.result()
                            except Exception as e:
                                self.get_logger().info(
                                    'Service call failed %r' % (e,))
                            else:
                                pass
                                # self.get_logger().info('Result of tablet plan:')
                                # self.get_logger().info(str(response.tablet_position))
                                # self.get_logger().info(str(response.tablet_orientation))
                            break

                    # update ik
                    # tablet_pose = self.planner.in_front_of_eyes(human=self.human)
                    tablet_pose = sp.SE3(R.from_quat(response.tablet_orientation).as_matrix(), response.tablet_position)

                    ik_soln, _ = self.planner.ik(tablet_pose)
                    msg = String()
                    msg.data = json.dumps(ik_soln)
                    self.get_logger().info("publishing " + msg.data)
                    self.pub_tablet_goal.publish(msg)

                    # clear buffer
                    self.human.pose_estimate.clear_estimates()

            except Exception as ex:
                print(ex)
            
            rclpy.spin_once(self, timeout_sec=0.1)

def main():
    rclpy.init()
    ShowTabletNode().main()

if __name__ == '__main__':
    main()