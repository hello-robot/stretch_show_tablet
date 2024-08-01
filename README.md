# Stretch Show Tablet

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31012/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Tested with python 3.10 on a [Stretch RE3](https://hello-robot.com/stretch-3-product) with a tablet tool. **Development Notice:** The code in this repo is a work-in-progress. The code in this repo may be unstable, since we are actively conducting development. Since we have performed limited testing, you may encounter unexpected behaviors.

# Overview

![Show Tablet](res/show_tablet.png)

**Stretch Show Tablet** contains ROS2 packages for autonomously presenting a tablet to a human. These packages are integrated into the experimental branch `vinitha/tablet_placement` of the [Stretch Web Teleoperation](https://github.com/hello-robot/stretch_web_teleop) repository. Support for standalone code functionality is in progress.

## Contents

This repository contains two ROS2 packages: `stretch_show_tablet` and `stretch_show_tablet_interfaces`. Inverse kinematics are handled using the [Pinocchio IK solver](https://github.com/hello-robot/stretch_web_teleop/blob/master/stretch_web_teleop_helpers/pinocchio_ik_solver.py) from `stretch_web_teleop`.

### stretch_show_tablet

This package contains the ROS Actions, planning framework, and human perception logic required to perceive humans and plan tablet showing locations.

- `plan_tablet_pose_service.py` accepts a `PlanTabletPose` service request and returns the location of a tablet relative to a human for easy viewing.
- `show_tablet_server.py` exposes a ROS Action of type `ShowTablet`. This action estimates a human's pose, uses the pose estimate to plan a tablet location for easy viewing, and then sends motion commands to the `stretch_driver` to move the robot's end effector to the target location.

### stretch_show_tablet_interfaces

This package contains the service and action message definitions required by the `stretch_show_tablet` action servers and services.

# Installation

Clone this repository into your ROS2 workspace, such as:

```bash
cd ~/ament_ws/src
git clone https://github.com/hello-lamsey/stretch_show_tablet
```

Install ROS dependencies and python dependencies using the following commands:

## ROS2 Packages + Dependencies Installation

On your robot:

```bash
cd ~/ament_ws
colcon build --packages-select stretch_show_tablet stretch_show_tablet_interfaces
```

TODO: add `rosdep` install instructions

## Python Dependencies

On your robot:

```bash
cd ~/ament_ws/src/stretch_show_tablet
pip install requirements.txt
```

# Examples

There are several ways to interact with the **Stretch Show Tablet** action server.

## Web Teleoperation Integration

The best way to interact with the **Stretch Show Tablet** action server is through the [Stretch Web Teleop](https://github.com/hello-robot/stretch_web_teleop) application.

TODO: Add details for how to enable tablet showing in web teleop!

## Demo: Command Line Interface

In two terminals, run the following commands:

Terminal one: `ros2 launch stretch_show_tablet demo_show_tablet.launch.py`

Terminal two: `ros2 run stretch_show_tablet demo_show_tablet`

The `demo_show_tablet` node will send a `ShowTablet` action request to the `show_tablet_server` action server. The action server will estimate the human's pose and plan a tablet location for easy viewing. Be sure to enable the pose estimator in the demo node CLI menu!

## Sending a ShowTablet Action Request

It is also possible to directly interact with the `show_tablet_server` action server using the `ros2 action send_goal` command. For example, to send a `ShowTablet` action request to the `show_tablet_server` action server, run the following commands one at a time:

```bash
ros2 launch stretch_show_tablet demo_show_tablet.launch.py
ros2 service call /toggle_body_pose_estimator std_srvs/srv/SetBool "{data: True}"
ros2 action send_goal /show_tablet stretch_show_tablet_interfaces/action/ShowTablet number_of_pose_estimates:\ 10
```

# Notes

The **Stretch Show Tablet** action uses the human pose estimator from [Stretch Deep Perception](https://github.com/hello-robot/stretch_ros2/tree/humble/stretch_deep_perception) inside of Stretch ROS2.

\[Insert image of how the human's coordinate system is generated\]
