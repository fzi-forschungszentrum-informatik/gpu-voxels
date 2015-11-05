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
* \author  Felix Mauch <mauch@fzi.de>
* \date    2015-05-07
*
*/
//----------------------------------------------------------------------

#ifndef GPU_VOXELS_VISUALIZATION_SHAREDMEMORYMANAGERVOXELLISTS_H
#define GPU_VOXELS_VISUALIZATION_SHAREDMEMORYMANAGERVOXELLISTS_H

#include <gpu_voxels/voxellist/VoxelList.h>
#include <gpu_voxels/vis_interface/VisualizerInterface.h>


namespace gpu_voxels {
namespace visualization {

class SharedMemoryManager;

class SharedMemoryManagerVoxelLists
{
public:
  SharedMemoryManagerVoxelLists();
  ~SharedMemoryManagerVoxelLists();

  uint32_t getNumberOfVoxelListsToDraw();

  bool getVoxelListName(std::string& map_name, const uint32_t index);

  bool getVisualizationData(Cube*& cubes, uint32_t& size,const uint32_t index);
  void setBufferSwappedToFalse(const uint32_t index);
  bool hasBufferSwapped(const uint32_t index);

private:
  SharedMemoryManager* shmm;
};


} //end of namespace visualization
} //end of namespace gpu_voxels

#endif // GPU_VOXELS_VISUALIZATION_SHAREDMEMORYMANAGERVOXELLISTS_H
