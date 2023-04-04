# ROS_interface
This is the ROS interface of real-world experiments.  
## Installion
Check the CMakeLists.txt and run it on Ubuntu.  
## Usage
You need to check the base coordinate system before the first run.  
Change the "gt_mode" in DefaultConfiguration.yaml.  
### Mode 1
The system will traverse all candidate views and save the ground truth point cloud.  
Make sure there are 32 views moved so that the candidate views are all reached by the robot.
### Mode 0
You have to run Mode 1 first and then the evaluation can go with the ground truth.  
The system will reconstuct the object with the setup in DefaultConfiguration.yaml.  
### Change in setup
Most is similar to our simulator.  
"height_of_ground" is to filter out the tabletop.  
"min_z_table" is the bottom height of the BBX.  
"move_dis_pre_point" is the movement length per action.  
"height_to_filter_arm" is to filter out the points above this value.  