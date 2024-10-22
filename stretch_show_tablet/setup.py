import os
from glob import glob

from setuptools import setup

package_name = "stretch_show_tablet"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/rviz", glob("rviz/*")),
        (
            os.path.join("share", package_name, "action"),
            glob("action/*.action"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Hello Robot Inc.",
    maintainer_email="support@hello-robot.com",
    description="The stretch_show_tablet package",
    license="Apache 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # 'detect_objects = stretch_show_tablet.detect_objects:main',
            "detect_faces = stretch_show_tablet.detect_faces:main",
            "detect_faces_gripper = stretch_show_tablet.detect_faces_gripper:main",
            "track_head = stretch_show_tablet.track_head:main",
            # 'detect_nearest_mouth = stretch_show_tablet.detect_nearest_mouth:main',
            "detect_body_landmarks = stretch_show_tablet.detect_body_landmarks:main",
            "record_test_data = stretch_show_tablet.record_test_data:main",
            "show_tablet_test = stretch_show_tablet.show_tablet_test:main",
            "stretch_main = stretch_show_tablet.stretch_main:main",
            "show_tablet_server = stretch_show_tablet.show_tablet_server:main",
            "estimate_pose_server = stretch_show_tablet.estimate_pose_server:main",
            "plan_tablet_pose_service = stretch_show_tablet.plan_tablet_pose_service:main",
            "track_head_server = stretch_show_tablet.track_head_server:main",
            "planner_service_test = stretch_show_tablet.planner_service_test:main",
            "demo_show_tablet = stretch_show_tablet.demo_show_tablet_action:main",
        ],
    },
)
