cmake_minimum_required(VERSION 3.0.2)
project(ma_scvp_real)

set(CMAKE_CXX_STANDARD 14)
add_compile_options(
  -O3
)

set (CMAKE_CXX_FLAGS "-g")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  pcl_ros
  tf2_ros
  tf2_sensor_msgs
  cv_bridge
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
  eigen_conversions
  octomap_msgs
  pcl_conversions
  nav_msgs
)

## System dependencies are found with CMake's conventions
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(octomap REQUIRED)
find_package(Boost REQUIRED)

catkin_package(
  INCLUDE_DIRS include
#  LIBRARIES ma_scvp_real
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  /home/opt/gurobi1000/linux64/include/
)

link_directories(/home/opt/gurobi1000/linux64/lib/)

add_executable(ma_scvp_real_node
  src/ros_interface.cpp
)

add_dependencies(ma_scvp_real_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(ma_scvp_real_node PUBLIC
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  pthread
  gurobi100
  gurobi_c++
  octomap
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
