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
This process needs to be done twice to obtain a correct bouding box.
### Mode 0
You have to run Mode 1 first and then the evaluation can go with the ground truth.  
The system will reconstuct the object with the setup in DefaultConfiguration.yaml.  