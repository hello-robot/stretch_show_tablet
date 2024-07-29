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

    d435i_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(stretch_core_path, "launch"),
                "/stretch_realsense.launch.py",
            ]
        )
    )

    detect_body_landmarks = Node(
        package="stretch_show_tablet",
        executable="detect_body_landmarks",
        output="screen",
    )

    detect_faces = Node(
        package="stretch_show_tablet",
        executable="detect_faces",
        output="screen",
    )

    record_data = Node(
        package="stretch_show_tablet",
        executable="record_test_data",
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
            d435i_launch,
            detect_body_landmarks,
            detect_faces,
            record_data,
            # rviz_node,  # uncomment if not running headless
        ]
    )
