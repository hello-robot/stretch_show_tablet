import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    stretch_core_path = get_package_share_directory('stretch_core')
    show_tablet_path = get_package_share_directory('stretch_tablet')
    stretch_deep_perception_path = get_package_share_directory('stretch_deep_perception')

    stretch_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([str(stretch_core_path), '/launch/stretch_driver.launch.py']),
        launch_arguments={'mode': 'position', 'broadcast_odom_tf': 'True', 'fail_out_of_range_goal': 'False'}.items(),
    )

    # d435i_launch = IncludeLaunchDescription(
    #       PythonLaunchDescriptionSource([os.path.join(
    #            stretch_core_path, 'launch'),
    #            '/stretch_realsense.launch.py'])
    #       )

    multi_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            show_tablet_path, 'launch'),
            '/multi_camera.launch.py'])
    )

    detect_body_landmarks = Node(
        package='stretch_tablet',
        executable='detect_body_landmarks',
        output='screen',
        )

    estimate_pose_server = Node(
        package='stretch_tablet',
        executable='estimate_pose_server',
        output='screen',
    )

    rviz_config_path = os.path.join(stretch_deep_perception_path, 'rviz', 'body_landmark_detection.rviz')

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen',
        )

    return LaunchDescription([
        stretch_driver,
        multi_camera_launch,
        detect_body_landmarks,
        estimate_pose_server,
        # rviz_node,  # uncomment if not running headless
        ])