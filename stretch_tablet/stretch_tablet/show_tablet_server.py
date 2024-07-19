import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse, ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.time import Time

from std_msgs.msg import String
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from geometry_msgs.msg import PoseStamped, Point

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import sophuspy as sp
import numpy as np
from scipy.spatial.transform import Rotation as R

from stretch_tablet_interfaces.srv import PlanTabletPose
from stretch_tablet_interfaces.action import ShowTablet

from enum import Enum
import json

from stretch_tablet.utils import load_bad_json_data
from stretch_tablet.utils_ros import generate_pose_stamped
from stretch_tablet.human import Human, HumanPoseEstimate

# testing
import time

# helpers
JOINT_NAME_SHORT_TO_FULL = {
    "base": "rotate_mobile_base",
    "lift": "joint_lift",
    "arm_extension": "wrist_extension",
    "yaw": "joint_wrist_yaw",
    "pitch": "joint_wrist_pitch",
    "roll": "joint_wrist_roll",
}

JOINT_NAME_FULL_TO_SHORT = {v: k for k, v in JOINT_NAME_SHORT_TO_FULL.items()}

PI_2 = np.pi/2.

def enforce_limits(value, min_value, max_value):
    return min([max([min_value, value]), max_value])

def enforce_joint_limits(pose: dict) -> dict:
    pose["base"] = enforce_limits(pose["base"], -PI_2, PI_2)
    pose["lift"] = enforce_limits(pose["lift"], 0.25, 1.1)
    pose["arm_extension"] = enforce_limits(pose["arm_extension"], 0.02, 0.45)
    pose["yaw"] = enforce_limits(pose["yaw"], -np.deg2rad(60.), PI_2)
    pose["pitch"] = enforce_limits(pose["pitch"], -PI_2, 0.2)
    # pose["roll"] = enforce_limits(pose["roll"], -PI_2, PI_2)
    pose["roll"] = 0.

    return pose

# classes
class ShowTabletState(Enum):
    IDLE = 0
    ESTIMATE_HUMAN_POSE = 1
    PLAN_TABLET_POSE = 2
    NAVIGATE_BASE = 3  # not implemented yet
    MOVE_ARM_TO_TABLET_POSE = 4
    END_INTERACTION = 5
    EXIT = 98
    ABORT = 99
    ERROR = -1

class ShowTabletActionServer(Node):
    def __init__(self):
        super().__init__('show_tablet_action_server')
        self._action_server = ActionServer(
            self,
            ShowTablet,
            'show_tablet',
            self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            cancel_callback=self.callback_action_cancel)
        
        # pub
        self.pub_valid_estimates = self.create_publisher(
            String,
            "/human_estimates/latest_body_pose",
            qos_profile=1
        )
        
        # sub
        self.sub_body_landmarks = self.create_subscription(
            String,
            "/body_landmarks/landmarks_3d",
            callback=self.callback_body_landmarks,
            qos_profile=1
        )

        # srv
        self.srv_plan_tablet_pose = self.create_client(
            PlanTabletPose, 'plan_tablet_pose')
        
        # motion
        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/stretch_controller/follow_joint_trajectory",
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # tf
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(buffer=self.tf_buffer, node=self)

        # guts
        self.executor = MultiThreadedExecutor()
        
        # state
        self.feedback = ShowTablet.Feedback()
        self.goal_handle = None
        self.human = Human()
        self.abort = False
        self.result = ShowTablet.Result()

        self._pose_history = []

        self._robot_joint_target = None
        self._pose_estimator_enabled = False

        # config
        self._robot_move_time_s = 4.
        self._n_poses = 10

    # helpers
    def now(self) -> Time:
        return self.get_clock().now().to_msg()
    
    def get_pose_history(self, max_age=float("inf")):
        # TODO: check if this boi is stale
        return [p for (p, t) in self._pose_history]

    def set_human_camera_pose(self, world2camera_pose: PoseStamped):
        # unpack camera pose
        camera_pos = world2camera_pose.pose.position
        camera_ori = world2camera_pose.pose.orientation
        camera_position = np.array([camera_pos.x, camera_pos.y, camera_pos.z])
        camera_orientation = R.from_quat([camera_ori.x, camera_ori.y, camera_ori.z, camera_ori.w]).as_matrix()
        world2camera_pose = sp.SE3(camera_orientation, camera_position)
        self.human.pose_estimate.set_camera_pose(world2camera_pose)

    def build_tablet_pose_request(self):
        request = PlanTabletPose.Request()
        human = self.human

        if not self.human.pose_estimate.is_body_populated():
            self.get_logger().error("ShowTabletActionServer::build_tablet_pose_request: self.human empty!")
            return request

        # extract request info
        body_string = json.dumps(human.pose_estimate.body_estimate)
        camera_pose = human.pose_estimate.get_camera_pose()
        camera_position = camera_pose.translation()
        camera_orientation = R.from_matrix(camera_pose.rotationMatrix()).as_quat()

        # construct request
        request.human_joint_dict = body_string
        request.camera_pose = generate_pose_stamped(camera_position, camera_orientation, self.now())
        request.robot_pose = generate_pose_stamped([0.,0.,0.],[0.,0.,0.,1.], self.now())

        return request
    
    def lookup_camera_pose(self) -> PoseStamped:
        msg = PoseStamped()

        try:
            t = self.tf_buffer.lookup_transform(
                    # "odom",
                    "base_link",
                    "camera_color_optical_frame",
                    rclpy.time.Time())
            
            pos = Point(x=t.transform.translation.x,
                        y=t.transform.translation.y,
                        z=t.transform.translation.z)
            msg.pose.position = pos
            msg.pose.orientation = t.transform.rotation
            msg.header.stamp = t.header.stamp

        except Exception as e:
            self.get_logger().error(str(e))
            self.get_logger().error("EstimatePoseActionServer::lookup_camera_pose: returning empty pose!")
        
        return msg
    
    def destroy(self):
        self._action_server.destroy()
        super().destroy_node()

    def cleanup(self):
        # TODO: implement
        pass

    def __push_to_pose_history(self, pose_estimate: HumanPoseEstimate):
        entry = (pose_estimate, self.now())
        self._pose_history.append(entry)

        while len(self._pose_history) > self._n_poses:
            self._pose_history.pop(0)

    def __move_to_pose(self, joint_dict: dict, blocking: bool=True):
        joint_names = [k for k in joint_dict.keys()]
        joint_positions = [v for v in joint_dict.values()]

        # build message
        trajectory = FollowJointTrajectory.Goal()
        trajectory.trajectory.joint_names = [JOINT_NAME_SHORT_TO_FULL[j] for j in joint_names]
        trajectory.trajectory.points = [JointTrajectoryPoint()]
        trajectory.trajectory.points[0].positions = [p for p in joint_positions]
        trajectory.trajectory.points[0].time_from_start = Duration(seconds=self._robot_move_time_s).to_msg()

        future = self.arm_client.send_goal_async(trajectory)
        rclpy.spin_until_future_complete(self, future, executor=self.executor)
        if blocking:
            goal_handle = future.result().get_result_async()
            rclpy.spin_until_future_complete(self, goal_handle, executor=self.executor)

    def __present_tablet(self, joint_dict: dict):
        pose_tuck = {
            "arm_extension": 0.05,
            "yaw": 0.
        }

        if "base" in joint_dict.keys():
            pose_base = {
                "base": joint_dict["base"],
                "lift": joint_dict["lift"],
            }
        else:
            pose_base = {
                "lift": joint_dict["lift"]
            }

        pose_arm = {
            "arm_extension": joint_dict["arm_extension"],
        }

        pose_wrist = {
            "yaw": joint_dict["yaw"],
            "pitch": joint_dict["pitch"],
            "roll": joint_dict["roll"],
        }

        # sequence
        self.__move_to_pose(pose_tuck, blocking=True)
        self.__move_to_pose(pose_base, blocking=True)
        self.__move_to_pose(pose_arm, blocking=True)
        self.__move_to_pose(pose_wrist, blocking=True)

    # callbacks
    def callback_body_landmarks(self, msg: String):
        if self._pose_estimator_enabled:
            # config
            necessary_keys = [
                "nose",
                "neck",
                "right_shoulder",
                "left_shoulder"
            ]
        
            # Load data
            msg_data = msg.data.replace("\"", "")
            if msg_data == "None" or msg_data is None:
                return

            data = load_bad_json_data(msg_data)

            if data is None or data == "{}":
                return
            
            # populate pose
            latest_pose = HumanPoseEstimate()
            latest_pose.set_body_estimate(data)

            # get visible keypoints
            pose_keys = latest_pose.body_estimate.keys()

            # check if necessary keys visible
            can_see = True
            for key in necessary_keys:
                if key not in pose_keys:
                    self.get_logger().info("cannot see key joints!")
                    can_see = False
                    continue  # for
            
            if not can_see:
                return
            
            # publish pose and add to pose history
            # self.pub_valid_estimates.publish(String(data=msg_data))
            self.__push_to_pose_history(latest_pose)

            average_pose = HumanPoseEstimate.average_pose_estimates(self.get_pose_history())
            pose_str = average_pose.get_body_estimate_string()
            self.pub_valid_estimates.publish(String(data=pose_str))

    # action callbacks
    # def callback_action_success(self):
    #     self.cleanup()
    #     self.result.status = ShowTablet.Result.STATUS_SUCCESS
    #     return ShowTablet.Result()

    def callback_action_cancel(self, goal_handle):
        self.cleanup()
        self.result.status = ShowTablet.Result.STATUS_CANCELED
        return CancelResponse.ACCEPT
        # return ShowTablet.Result(result=-1)

    # def callback_action_error(self):
    #     self.cleanup()
    #     self.result.status = ShowTablet.Result.STATUS_ERROR
    #     return ShowTablet.Result()

    # states
    def state_idle(self):
        if self.abort or self.goal_handle.is_cancel_requested:
            return ShowTabletState.ABORT

        if False:
            return ShowTabletState.IDLE
        
        return ShowTabletState.ESTIMATE_HUMAN_POSE
    
    def state_estimate_pose(self):
        if self.abort or self.goal_handle.is_cancel_requested:
            return ShowTabletState.ABORT

        self._pose_estimator_enabled = True

        while rclpy.ok():
            if len(self._pose_history) >= self._n_poses:
                break

        self._pose_estimator_enabled = False

        self.human.pose_estimate = HumanPoseEstimate.average_pose_estimates(self.get_pose_history())
        self.set_human_camera_pose(self.lookup_camera_pose())

        return ShowTabletState.PLAN_TABLET_POSE

    def state_plan_tablet_pose(self):
        if self.abort or self.goal_handle.is_cancel_requested:
            return ShowTabletState.ABORT
        
        # if not self.human.pose_estimate.is_populated():
            # Human not seen
            # return ShowTabletState.LOOK_AT_HUMAN
        
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
                break

        if response.success:
            # get planner result
            joint_names = response.robot_ik_joint_names
            joint_positions = response.robot_ik_joint_positions

            # process output
            joint_dict = {n: p for n, p in zip(joint_names, joint_positions)}
            joint_dict = enforce_joint_limits(joint_dict)

            # robot cannot do small moves
            if abs(joint_dict["base"]) < 0.2:
                joint_dict.pop("base")

            # debug
            self.get_logger().info("JOINT CONFIG:")
            self.get_logger().info(json.dumps(joint_dict, indent=2))

            self._robot_joint_target = joint_dict

        # return ShowTabletState.NAVIGATE_BASE
        return ShowTabletState.MOVE_ARM_TO_TABLET_POSE

    def state_navigate_base(self):
        if self.abort or self.goal_handle.is_cancel_requested:
            return ShowTabletState.ABORT

        # TODO: Implement this state
        # Drive the robot to a human-centric location

        return ShowTabletState.MOVE_ARM_TO_TABLET_POSE
    
    def state_move_arm_to_tablet_pose(self):
        if self.abort or self.goal_handle.is_cancel_requested:
            return ShowTabletState.ABORT

        target = self._robot_joint_target
        self.__present_tablet(target)
        self.get_logger().info('Finished move_to_pose')

        return ShowTabletState.EXIT

    def state_end_interaction(self):
        if self.abort or self.goal_handle.is_cancel_requested:
            return ShowTabletState.ABORT

        return ShowTabletState.EXIT
    
    def state_abort(self):
        # TODO: stop all motion
        return ShowTabletState.EXIT

    def state_exit(self):
        self.result.status = ShowTablet.Result.STATUS_SUCCESS
        return ShowTabletState.EXIT

    def run_state_machine(self, test:bool=False):
        state = self.state_idle()

        while rclpy.ok():
            # check abort
            if state == ShowTabletState.ABORT:
                state = self.state_abort()
                break

            # main state machine
            elif state == ShowTabletState.IDLE:
                state = self.state_idle()
            elif state == ShowTabletState.ESTIMATE_HUMAN_POSE:
                state = self.state_estimate_pose()
            elif state == ShowTabletState.PLAN_TABLET_POSE:
                state = self.state_plan_tablet_pose()
            # elif state == ShowTabletState.NAVIGATE_BASE:
            #     state = self.state_navigate_base()
            elif state == ShowTabletState.MOVE_ARM_TO_TABLET_POSE:
                state = self.state_move_arm_to_tablet_pose()
            # elif state == ShowTabletState.TRACK_FACE:
            #     state = self.state_track_face()
            elif state == ShowTabletState.END_INTERACTION:
                state = self.state_end_interaction()
            elif state == ShowTabletState.EXIT:
                state = self.state_exit()
                break
            else:
                state = ShowTabletState.IDLE

            self.feedback.current_state = state.value
            self.goal_handle.publish_feedback(self.feedback)
            if test:
                time.sleep(0.25)
        
        return self.result

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing Show Tablet...')
        self.goal_handle = goal_handle

        # load body pose
        # pose_estimate = load_bad_json_data(goal_handle.request.human_joint_dict)
        # self.human.pose_estimate.set_body_estimate(pose_estimate)
        # self.set_human_camera_pose(goal_handle.request.camera_pose)

        # self.get_logger().info(str(self.human.pose_estimate.get_body_world()))
        # self.get_logger().info(str(self.human.pose_estimate.get_camera_pose()))
        
        final_result = self.run_state_machine(test=True)

        # get result
        if final_result.status == ShowTablet.Result.STATUS_SUCCESS:
            goal_handle.succeed()
        elif final_result.status == ShowTablet.Result.STATUS_ERROR:
            goal_handle.abort()
        elif final_result.status == ShowTablet.Result.STATUS_CANCELED:
            goal_handle.canceled()
        else:
            raise ValueError
        
        self.get_logger().info(str(final_result))
        return final_result


def main(args=None):
    rclpy.init(args=args)

    # Use a MultiThreadedExecutor to enable processing goals concurrently
    executor = MultiThreadedExecutor()
    show_tablet_action_server = ShowTabletActionServer()
    rclpy.spin(show_tablet_action_server, executor=executor)
    show_tablet_action_server.destroy()


if __name__ == '__main__':
    main()