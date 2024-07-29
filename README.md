# Description

Work in progress - showing tablet code

# Installation

## ROS2 Packages Installation

On your robot:

```bash
cd ~/ament_ws/src
git clone https://github.com/hello-lamsey/stretch_show_tablet
cd ~/ament_ws
colcon build --packages-select stretch_tablet stretch_tablet_interfaces
```

## Python Dependencies

`pip install sophuspy`

# Notes

There are some hardcoded paths in here that might need to be changed...

# How to run demo

First, position the robot's with the arm pointing towards you and the robot's head facing you. Then, run in three terminals:

```
ros2 launch show_tablet_test_launch.py
ros2 run stretch_tablet stretch_main
```

In the terminal running `stretch_main`, confirm the robot's pose printed to the terminal and then enter `y` to confirm the motion.
