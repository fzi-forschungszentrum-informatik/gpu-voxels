== Planning example combining OMPL and GPU-Voxels ==

This example uses GPU-Voxels data structures and collision checking to plan the movement of a 6DOF robot arm in OMPL.

== Requirements ==

* OMPL from ROS Kinetic
* GPU-Voxels built without PCL support, or linked against a C++11 compatible version like 1.8
* CUDA 8.0

== Setup ==

* build GPU-Voxels with C++11 enabled and install to gpu-voxels/export

 cd <gpu-voxels-path>
 # make sure SET(CMAKE_CXX_STANDARD 11) is uncommented in packages/gpu_voxels/CMakeLists.txt
 mkdir build
 cd build
 cmake -DCMAKE_DISABLE_FIND_PACKAGE_PCL=TRUE ..
 make && make install

* build planner

 cd <gpu-voxels-path>/gvl_ompl_planning
 mkdir build
 cd build
 cmake -DCMAKE_PREFIX_PATH=<gpu-voxels-path>/export ..
 make

== Running the planner ==

* Launch planner
 export GPU_VOXELS_MODEL_PATH=<gpu-voxels-path>/packages/gpu_voxels/models/
 # if libs are not found: export LD_LIBRARY_PATH=<gpu-voxels-path>/export/lib:$LD_LIBRARY_PATH
 ./gvl_ompl_planner

* Start Visualizer
 <gpu_voxels>/build/bin/gpu_voxels_visualizer
 

== When building GPU-Voxels with PCL 1.8.1 ==
* build PCL 1.8.1 from source                                                                                                                                              

* build GPU-Voxels with PCL 1.8.1
 # make sure SET(CMAKE_CXX_STANDARD 11) is uncommented in packages/gpu_voxels/CMakeLists.txt
 cd build
 cmake .. -DCMAKE_PREFIX_PATH=~/pcl-1.8.1/build:$CMAKE_PREFIX_PATH # use pcl version 1.8.1
 make && make install
 # run "bin/gpu_voxels_visualizer" after starting gvl_ompl_planning

* build gvl_ompl_planning
 cd gvl_ompl_planning
 mkdir build
 cd build
 export LD_LIBRARY_PATH=~/pcl-1.8.1/build/lib:$LD_LIBRARY_PATH # if you still have pcl1.7 installed
 cmake -DCMAKE_PREFIX_PATH=<gpu-voxels-path>/export:~/pcl-1.8.1/build/lib/ ..
 make
 ldd gvl_ompl_planner | grep -F "libpcl" # make sure there are no pcl 1.7 libraries linked in
 # if libs are not found: export LD_LIBRARY_PATH=<gpu-voxels-path>/export/lib:$LD_LIBRARY_PATH
 ./gvl_ompl_planner 

