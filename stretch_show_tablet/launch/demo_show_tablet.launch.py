import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    show_tablet_path = get_package_share_directory("stretch_show_tablet")
    stretch_core_path = get_package_share_directory("stretch_core")
    stretch_deep_perception_path = get_package_share_directory(
        "stretch_deep_perception"
    )

    stretch_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [str(stretch_core_path), "/launch/stretch_driver.launch.py"]
        ),
        launch_arguments={
            "mode": "position",
            "broadcast_odom_tf": "True",
            "fail_out_of_range_goal": "False",
        }.items(),
    )

    multi_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(show_tablet_path, "launch"),
                "/multi_camera.launch.py",
            ]
        )
    )

    # deep perception
    detect_body_landmarks = Node(
        package="stretch_show_tablet",
        executable="detect_body_landmarks",
        output="screen",
    )

    # actions and services
    show_tablet_server = Node(
        package="stretch_show_tablet",
        executable="show_tablet_server",
        output="screen",
    )

    planner_service = Node(
        package="stretch_show_tablet",
        executable="plan_tablet_pose_service",
        output="screen",
    )

    # NOTE: doesn't work for user input when launched from a launch file, lame
    # nodes
    # demo_node = Node(
    #     package="stretch_show_tablet",
    #     executable="demo_show_tablet",
    #     emulate_tty=True,  # for screen output
    #     output="screen"
    # )

    rviz_config_path = os.path.join(
        stretch_deep_perception_path, "rviz", "body_landmark_detection.rviz"
    )

    rviz_node = Node(  # noqa: F841
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_config_path],
        output="screen",
    )

    return LaunchDescription(
        [
            stretch_driver,
            multi_camera_launch,
            detect_body_landmarks,
            show_tablet_server,
            planner_service,
            # rviz_node,  # uncomment if not running headless
        ]
    )
