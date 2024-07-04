import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node

from stretch_tablet_interfaces.action import ShowTablet

from enum import Enum
import time

class ShowTabletState(Enum):
    IDLE = 0
    LOOK_AT_HUMAN = 1
    PLAN_TABLET_POSE = 2
    NAVIGATE_BASE = 3
    MOVE_ARM_TO_TABLET_POSE = 4
    TRACK_FACE = 5
    END_INTERACTION = 6
    EXIT = 99

class ShowTabletActionServer(Node):
    def __init__(self):
        super().__init__('show_tablet_action_server')
        self._action_server = ActionServer(
            self,
            ShowTablet,
            'show_tablet',
            self.execute_callback)
        
        # state
        self.feedback = ShowTablet.Feedback()
        self.abort = False

    # states
    def state_idle(self):
        if self.abort:
            return ShowTabletState.EXIT

        if False:
            return ShowTabletState.IDLE
        
        return ShowTabletState.LOOK_AT_HUMAN

    def state_look_at_human(self):
        if self.abort:
            return ShowTabletState.EXIT

        return ShowTabletState.PLAN_TABLET_POSE

    def state_plan_tablet_pose(self):
        if self.abort:
            return ShowTabletState.EXIT

        return ShowTabletState.NAVIGATE_BASE

    def state_navigate_base(self):
        if self.abort:
            return ShowTabletState.EXIT

        return ShowTabletState.MOVE_ARM_TO_TABLET_POSE
    
    def state_move_arm_to_tablet_pose(self):
        if self.abort:
            return ShowTabletState.EXIT

        return ShowTabletState.TRACK_FACE

    def state_track_face(self):
        if self.abort:
            return ShowTabletState.EXIT

        return ShowTabletState.END_INTERACTION

    def state_end_interaction(self):
        if self.abort:
            return ShowTabletState.EXIT

        return ShowTabletState.EXIT

    def state_exit(self):
        return ShowTabletState.EXIT

    def run_state_machine(self, goal_handle, test:bool=False):
        state = self.state_idle()

        while rclpy.ok():
            if state == ShowTabletState.IDLE:
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
                return
            else:
                state = ShowTabletState.IDLE

            self.feedback.current_state = state.value
            goal_handle.publish_feedback(self.feedback)
            if test:
                time.sleep(0.25)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing Show Tablet...')
        
        self.run_state_machine(goal_handle, test=True)

        # get result
        result = ShowTablet.Result()
        goal_handle.succeed()
        return result


def main(args=None):
    rclpy.init(args=args)

    show_tablet_action_server = ShowTabletActionServer()

    rclpy.spin(show_tablet_action_server)


if __name__ == '__main__':
    main()