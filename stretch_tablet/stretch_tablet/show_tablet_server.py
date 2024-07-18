import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse, ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from std_msgs.msg import String
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from action_msgs.srv import CancelGoal

import sophuspy as sp
import numpy as np
from scipy.spatial.transform import Rotation as R

from stretch_tablet_interfaces.srv import PlanTabletPose
from stretch_tablet_interfaces.action import ShowTablet

from enum import Enum
import json

from stretch_tablet.utils import load_bad_json_data
from stretch_tablet.utils_ros import generate_pose_stamped
from stretch_tablet.human import Human

import threading

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
    PLAN_TABLET_POSE = 1
    NAVIGATE_BASE = 2  # not implemented yet
    MOVE_ARM_TO_TABLET_POSE = 3
    END_INTERACTION = 4
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
        
        # sub
        self.sub_joint_state = self.create_subscription(
            JointState,
            "/joint_states",
            callback=self.callback_joint_state,
            qos_profile=1
        )

        # srv
        self.srv_plan_tablet_pose = self.create_client(
            PlanTabletPose, 'plan_tablet_pose')
        
        self.srv_cancel_move = self.create_client(
            CancelGoal, '/show_tablet/_action/cancel_goal'
        )
        
        # motion
        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/stretch_controller/follow_joint_trajectory",
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # guts
        # TODO: get rid?
        # self.executor = MultiThreadedExecutor()
        
        # state
        self.feedback = ShowTablet.Feedback()
        self.goal_handle = None
        self.human = Human()
        self.abort = False
        self.result = ShowTablet.Result()
        self._robot_joint_target = None
        self._joint_state = JointState()

        # config
        self._robot_move_time_s = 4.

    # helpers
    def now(self):
        return self.get_clock().now().to_msg()

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
    
    def destroy(self):
        self._action_server.destroy()
        super().destroy_node()

    def cleanup(self):
        self.__stop_motion()

    def __is_cancelled(self):
        cancel_requested = False
        if self.goal_handle is not None:
            cancel_requested = self.goal_handle.is_cancel_requested
        
        return cancel_requested or self.abort

    def __move_to_pose(self, joint_dict: dict, blocking: bool=True):
        joint_names = [k for k in joint_dict.keys()]
        joint_positions = [v for v in joint_dict.values()]

        # build message
        trajectory = FollowJointTrajectory.Goal()
        trajectory.trajectory.joint_names = [JOINT_NAME_SHORT_TO_FULL[j] for j in joint_names]
        trajectory.trajectory.points = [JointTrajectoryPoint()]
        trajectory.trajectory.points[0].positions = [p for p in joint_positions]
        trajectory.trajectory.points[0].time_from_start = Duration(seconds=self._robot_move_time_s).to_msg()

        # send goal + wait until finished while checking for cancel
        rate = self.create_rate(10.)
        future = self.arm_client.send_goal_async(trajectory)

        self.get_logger().info('here 0')
        while rclpy.ok() and not future.done():
            self.get_logger().info('inside!')
            rate.sleep()

        self.get_logger().info('here 1')
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("ShowTabletActionServer::__move_to_pose: goal rejected :(")
            return
    
        self.get_logger().info('here 2')
        goal_future = goal_handle.get_result_async()
        while rclpy.ok() and not goal_future.done():
            self.get_logger().info('here in')
            if self.__is_cancelled():
                self.__stop_motion()
                return
            rate.sleep()
        
        # rclpy.spin_until_future_complete(self, future, executor=self.executor)  # TODO: kill

        # if blocking:
        #     spin_thread = threading.Thread(target=rclpy.spin, args=(self, self.executor))  # TODO: kill
        #     spin_thread.start()

        #     rate = self.create_rate(10.)
        #     goal_handle = future.result().get_result_async()
        #     while rclpy.ok() and (not goal_handle.done()):
        #         if self.__is_cancelled():
        #             self.__stop_motion()
        #             return
        #         rate.sleep()
        
        #     spin_thread.join()

                
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
        for pose in [pose_tuck, pose_base, pose_arm, pose_wrist]:
            if self.__is_cancelled():
                return
            
            self.__move_to_pose(pose, blocking=True)

    def __stop_motion(self):
        self.srv_cancel_move.call_async(CancelGoal.Request())
        # js = self._joint_state
        # try:
        #     lift = js.position[js.name.index("joint_lift")]
        #     arm = sum([js.position[js.name.index("joint_arm_l" + str(i))] for i in range(4)])
        #     yaw = js.position[js.name.index("joint_wrist_yaw")]
        #     pitch = js.position[js.name.index("joint_wrist_pitch")]
        #     roll = js.position[js.name.index("joint_wrist_roll")]
        #     joint_dict = {
        #         "lift": lift,
        #         "arm_extension": arm,
        #         "yaw": yaw,
        #         "pitch": pitch,
        #         "roll": roll,
        #         "base": 0.
        #     }
        #     self.__move_to_pose(joint_dict, blocking=False)
        # except Exception as e:
        #     self.get_logger().error(str(e))

    # action callbacks
    # def callback_action_success(self):
    #     self.cleanup()
    #     self.result.status = ShowTablet.Result.STATUS_SUCCESS
    #     return ShowTablet.Result()

    def callback_action_cancel(self, goal_handle):
        self.cleanup()
        self.abort = True
        self.result.status = ShowTablet.Result.STATUS_CANCELED
        return CancelResponse.ACCEPT
        # return ShowTablet.Result(result=-1)

    # def callback_action_error(self):
    #     self.cleanup()
    #     self.result.status = ShowTablet.Result.STATUS_ERROR
    #     return ShowTablet.Result()

    def callback_joint_state(self, msg: JointState):
        self._joint_state = msg

    # states
    def state_idle(self):
        if self.__is_cancelled():
            return ShowTabletState.ABORT

        if False:
            return ShowTabletState.IDLE

        return ShowTabletState.PLAN_TABLET_POSE

    def state_plan_tablet_pose(self):
        if self.__is_cancelled():
            return ShowTabletState.ABORT
        
        # if not self.human.pose_estimate.is_populated():
            # Human not seen
            # return ShowTabletState.LOOK_AT_HUMAN
        
        plan_request = self.build_tablet_pose_request()
        future = self.srv_plan_tablet_pose.call_async(plan_request)

        # wait for srv to finish
        while rclpy.ok():
            rclpy.spin_once(self)

            if self.__is_cancelled():
                return ShowTabletState.ABORT
            
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
        
        if self.__is_cancelled():
            return ShowTabletState.ABORT

        # return ShowTabletState.NAVIGATE_BASE
        return ShowTabletState.MOVE_ARM_TO_TABLET_POSE

    def state_navigate_base(self):
        if self.__is_cancelled():
            return ShowTabletState.ABORT

        # TODO: Implement this state
        # Drive the robot to a human-centric location

        return ShowTabletState.MOVE_ARM_TO_TABLET_POSE
    
    def state_move_arm_to_tablet_pose(self):
        if self.__is_cancelled():
            return ShowTabletState.ABORT

        target = self._robot_joint_target
        self.__present_tablet(target)
        self.get_logger().info('Finished move_to_pose')

        if self.__is_cancelled():
            return ShowTabletState.ABORT

        return ShowTabletState.EXIT

    def state_end_interaction(self):
        if self.__is_cancelled():
            return ShowTabletState.ABORT

        return ShowTabletState.EXIT
    
    def state_abort(self):
        self.cleanup()
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
        pose_estimate = load_bad_json_data(goal_handle.request.human_joint_dict)
        self.human.pose_estimate.set_body_estimate(pose_estimate)
        self.set_human_camera_pose(goal_handle.request.camera_pose)

        self.get_logger().info(str(self.human.pose_estimate.get_body_world()))
        self.get_logger().info(str(self.human.pose_estimate.get_camera_pose()))
        
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