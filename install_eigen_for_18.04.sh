#!/bin/bash

DEST=3rd-party/eigen-git-mirror
VERSION="3.3.7"

echo "Cloning Eigen3 $VERSION to $DEST ($(realpath $DEST))"
mkdir -p $DEST
git clone https://github.com/eigenteam/eigen-git-mirror.git -b $VERSION $DEST

echo "Add the Eigen3 root folder to CMAKE_PREFIX_PATH as follows:"
echo "export CMAKE_PREFIX_PATH=$(realpath ${DEST}):\$CMAKE_PREFIX_PATH"
