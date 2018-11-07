IcMaker - The build system for the interchange libraries
========================================================

Project icmaker is intended to be used for building the various interchange
libraries.

It is published under BSD license. See LICENSE for more details.

How to build IC libraries with IcMaker
======================================

IcMaker can be used as a standalone build system but also as supplementary
cmake functionality to support easier building inside other cmake workspaces
like for example ROS catkin workspaces.


Building Standalone
-------------------

The recommended full-featured setup is the _Standalone_ method:

### Setup

At first create a folder structure like this:
```
ic_workspace
  CMakeLists.txt [root]
  build
  packages
    CMakeLists.txt [packages]
```

Then clone the icmaker repository into the _ic_workspace_ folder:
```bash
cd ic_workspace
git clone git@github.com:fzi-forschungszentrum-informatik/icmaker.git
```

Clone all the packages you want to build into the packages folder, for example:
```bash
cd ic_workspace/packages
git clone git@github.com:fzi-forschungszentrum-informatik/icl_core.git
```

Fill the CMakeLists.txt with the following content:

CMakeLists.txt [root] definition:
```cmake
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

SET(ICMAKER_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/icmaker)

INCLUDE(${ICMAKER_DIRECTORY}/IcMaker.cmake)

ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/packages)
#ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/doc)

IF (EXISTS "${CMAKE_SOURCE_DIR}/script" AND IS_DIRECTORY "${CMAKE_SOURCE_DIR}/script")
  ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/script)
ENDIF ()
```


CMakeLists.txt [packages] definition:
```cmake
ADD_SUBDIRECTORY(icl_core)
# Add the packages you want to include in your build here.
```

### Building

Now you can build all libraries at once using:
```bash
cd ic_workspace/build
cmake ..
make -j4 install
```

The libraries and executables will automatically be installed into _ic_workspace/export_.

### Running

TODO: How to run an example executable.


Building in a Catkin Workspace
-----------------------------

The experimental setup is the _Catkin_ method:

### Setup

Create a catkin workspace or use an existing one.

Checkout icmaker and the packages you want to build:
```bash
cd catkin_ws/src
git clone git@github.com:fzi-forschungszentrum-informatik/icmaker.git
git clone git@github.com:fzi-forschungszentrum-informatik/icl_core.git
```

### Building and Running

You can build and run the libraries and executables like in any other package in
your catkin workspace.

### Known problems

If the package uses __direct dependencies to specific libraries__ using the
ICMAKER_DEPENDENCIES() macro, these dependencies will not be found when building
in a catkin workspace. If a library cannot be built because of missing
dependencies, you will have more success building the library with the
_Standalone_ method and use the installed libraries from _ic_workspace/export_
in your catkin or other projects.


Creating your own IC library
============================

A good start for creating your own IC library and using the icmaker macros is to look into an existing IC library. Especially ''icl_example'' is well-documented and meant to be a start point for new IC libraries.
