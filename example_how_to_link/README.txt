These files demonstrate on how to link against GPU-Voxels with CMake.
Also see the according doxygen page.

How-to:
Source your ROS environment, if you want URDF support, otherwise remove the two marked lines from this examples CMakeLists file (about KDL).

First build and install GPU-Voxels:
 cd /my-home/my/gvl/build/dir
 cmake .. -DCMAKE_INSTALL_PREFIX=/my-home/my/gvl/install/dir
 make && make install

Then build this example via:
 cd /this/examples/directory
 mkdir build
 cd build
 cmake .. -DCMAKE_PREFIX_PATH=/my-home/my/gvl/install/dir/
 make

Now you should be able to execute:
 ./gvl_linkage_test
and read about 8000 collisions.
