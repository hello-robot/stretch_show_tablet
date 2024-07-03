cmake_minimum_required(VERSION 3.5)
project(stretch_tablet)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

# Other dependencies
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/ShowTablet.action"
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()