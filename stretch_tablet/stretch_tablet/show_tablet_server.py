import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String

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

# testing
import time

class ShowTabletState(Enum):
    IDLE = 0
    LOOK_AT_HUMAN = 1
    PLAN_TABLET_POSE = 2
    NAVIGATE_BASE = 3
    MOVE_ARM_TO_TABLET_POSE = 4
    TRACK_FACE = 5
    END_INTERACTION = 6
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
        
        # srv
        self.srv_plan_tablet_pose = self.create_client(
            PlanTabletPose, 'plan_tablet_pose')
        
        # state
        self.feedback = ShowTablet.Feedback()
        self.goal_handle = None
        self.human = Human()
        self.abort = False
        self.result = ShowTablet.Result()

    # helpers
    def clear_landmarks(self):
        self.human.pose_estimate.clear_estimates()

    def build_tablet_pose_request(self):
        request = PlanTabletPose.Request()
        human = self.human

        if not self.human.pose_estimate.is_populated():
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
        # TODO: implement
        pass

    # callbacks
    def callback_body_landmarks(self, msg: String):
        msg_data = msg.data.replace("\"", "")
        if msg_data == "None" or msg_data is None:
            return

        data = load_bad_json_data(msg_data)

        if data is None or data == "{}":
            return
        self.human.pose_estimate.set_body_estimate(data)

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
        if self.abort:
            return ShowTabletState.ABORT
        
        if self.goal_handle.is_cancel_requested:
            return ShowTabletState.ABORT

        if False:
            return ShowTabletState.IDLE
        
        # return ShowTabletState.LOOK_AT_HUMAN
        return ShowTabletState.IDLE

    def state_look_at_human(self):
        if self.abort:
            return ShowTabletState.ABORT

        return ShowTabletState.PLAN_TABLET_POSE

    def state_plan_tablet_pose(self):
        if self.abort:
            return ShowTabletState.ABORT
        
        if not self.human.pose_estimate.is_populated():
            # Human not seen
            return ShowTabletState.LOOK_AT_HUMAN
        
        plan_request = self.build_tablet_pose_request()
        future = self.srv_plan_tablet_pose.call_async(plan_request)

        return ShowTabletState.NAVIGATE_BASE

    def state_navigate_base(self):
        if self.abort:
            return ShowTabletState.ABORT

        return ShowTabletState.MOVE_ARM_TO_TABLET_POSE
    
    def state_move_arm_to_tablet_pose(self):
        if self.abort:
            return ShowTabletState.ABORT

        return ShowTabletState.TRACK_FACE

    def state_track_face(self):
        if self.abort:
            return ShowTabletState.ABORT

        return ShowTabletState.END_INTERACTION

    def state_end_interaction(self):
        if self.abort:
            return ShowTabletState.ABORT

        return ShowTabletState.EXIT
    
    def state_abort(self):
        # stop all motion
        return ShowTabletState.EXIT

    def state_exit(self):
        self.result.status = ShowTablet.Result.STATUS_SUCCESS
        return ShowTabletState.EXIT

    def run_state_machine(self, test:bool=False):
        state = self.state_idle()

        while rclpy.ok():
            if state == ShowTabletState.ABORT:
                state = self.state_abort()
                break
            elif state == ShowTabletState.IDLE:
                state = self.state_idle()
            elif state == ShowTabletState.LOOK_AT_HUMAN:
                state = self.state_look_at_human()
            elif state == ShowTabletState.PLAN_TABLET_POSE:
                state = self.state_plan_tablet_pose()
            elif state == ShowTabletState.NAVIGATE_BASE:
                state = self.state_navigate_base()
            elif state == ShowTabletState.MOVE_ARM_TO_TABLET_POSE:
                state = self.state_move_arm_to_tablet_pose()
            elif state == ShowTabletState.TRACK_FACE:
                state = self.state_track_face()
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
        
        final_result = self.run_state_machine(test=True)

        # get result
        # result = ShowTablet.Result()
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