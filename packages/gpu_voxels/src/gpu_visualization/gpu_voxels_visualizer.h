// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
//
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE.txt in the top
// directory of the source code.
//
// © Copyright 2014 FZI Forschungszentrum Informatik, Karlsruhe, Germany
//
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Matthias Wagner
 * \date    2014-27-6
 *
 *  Wrapper for the Visualizer class.
 *
 *
 */

#ifndef GPU_VOXELS_VISUALIZER_H_
#define GPU_VOXELS_VISUALIZER_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>       /* time */
#include <math.h>
#include <iostream>
#include <fstream>
#include <cstdio>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>

#include <gpu_visualization/logging/logging_visualization.h>

// OpenGL Graphics includes
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/cuda_handling.h>
//#include <gpu_voxels/voxelmap/Voxel.h>
#include <gpu_visualization/Visualizer.h>

#include <icl_core_config/Config.h>
#include <gpu_voxels/voxelmap/VoxelMap.h>

#include <gpu_visualization/SharedMemoryManagerOctrees.h>
#include <gpu_visualization/SharedMemoryManagerVoxelMaps.h>

gpu_voxels::visualization::Visualizer* vis =
    new gpu_voxels::visualization::Visualizer();

void resizeFunctionWrapper(int32_t width, int32_t height)
{
  vis->resizeFunction(width, height);
}
void renderFunctionWrapper()
{
  vis->renderFunction();
}
void idleFunctionWrapper()
{
  vis->idleFunction();
}
void timerFunctionWrapper(int32_t value)
{
  vis->timerFunction(value, timerFunctionWrapper);
}
void cleanupFunctionWrapper(void)
{
  vis->cleanupFunction();
}
void keyboardFunctionWrapper(unsigned char key, int32_t x, int32_t y)
{
  vis->keyboardFunction(key, x, y);
}
void keyboardSpecialFunctionWrapper(int32_t key, int32_t x, int32_t y)
{
  vis->keyboardSpecialFunction(key, x, y);
}
void mouseMotionFunctionWrapper(int32_t xpos, int32_t ypos)
{
  vis->mouseMotionFunction(xpos, ypos);
}
void mousePassiveMotionFunctionWrapper(int32_t xpos, int32_t ypos)
{
  vis->mousePassiveMotionFunction(xpos, ypos);
}
void mouseClickFunctionWrapper(int32_t button, int32_t state, int32_t x, int32_t y)
{
  vis->mouseClickFunction(button, state, x, y);
}
void menuFunctionWrapper(int value)
{
  vis->menuFunction(value);
}

void runVisualisation(int32_t* argc, char* argv[]);

void createRightClickMenu();

void registerVoxelmapFromSharedMemory(uint32_t index);
void registerVoxellistFromSharedMemory(uint32_t index);
void registerOctreeFromSharedMemory(uint32_t index);
void registerPrimitiveArrayFromSharedMemory(uint32_t index);

uint32_t getNumberOfOctreesFromSharedMem();
uint32_t getNumberOfVoxelmapsFromSharedMem();
uint32_t getNumberOfVoxellistsFromSharedMem();
uint32_t getNumberOfPrimitiveArraysFromSharedMem();

#endif /* GPU_VOXELS_VISUALIZER_H_ */
