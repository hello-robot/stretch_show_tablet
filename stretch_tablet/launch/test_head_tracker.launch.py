import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
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

    stretch_main = Node(
        package="stretch_tablet",
        executable="stretch_main",
        output="screen",
    )

    d405_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [str(stretch_core_path), "/launch/d405_basic.launch.py"]
        )
    )

    detect_faces = Node(
        package="stretch_tablet",
        executable="detect_faces_gripper",
        output="screen",
    )

    track_head = Node(
        package="stretch_tablet",
        executable="track_head",
        output="screen",
    )

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
            stretch_main,
            d405_launch,
            detect_faces,
            track_head,
            # rviz_node,  # uncomment if not running headless
        ]
    )
