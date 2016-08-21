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
 * \date    2014-07-09
 *
 * \brief This class is for the management of the interprocess communication with the provider.
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_VISUALIZATION_SHAREDMEMORYMANAGEROCTREES_H_INCLUDED
#define GPU_VOXELS_VISUALIZATION_SHAREDMEMORYMANAGEROCTREES_H_INCLUDED

#include <boost/lexical_cast.hpp>
#include <gpu_voxels/vis_interface/VisualizerInterface.h>

namespace gpu_voxels {
namespace visualization {

// forward declaration so we don't have to include the header
// which contains boost code
class SharedMemoryManager;

class SharedMemoryManagerOctrees
{
public:
  SharedMemoryManagerOctrees();
  ~SharedMemoryManagerOctrees();
  uint32_t getNumberOfOctreesToDraw();
  std::string getNameOfOctree(const uint32_t index);
  bool getOctreeVisualizationData(Cube*& cubes, uint32_t& size,const uint32_t index);
  void setView(Vector3ui start_voxel, Vector3ui end_voxel);
  void setOctreeBufferSwappedToFalse(const uint32_t index);
  bool hasOctreeBufferSwapped(const uint32_t index);
  void setOctreeOccupancyThreshold(const uint32_t index, Probability threshold);
  bool getSuperVoxelSize(uint32_t & sdim);
  void setSuperVoxelSize(uint32_t sdim);
private:
  SharedMemoryManager* shmm;
}
;
} //end of namespace visualization
} //end of namespace gpu_voxels
#endif /* GPU_VOXELS_VISUALIZATION_SHAREDMEMORYMANAGEROCTREES_H_INCLUDED */
