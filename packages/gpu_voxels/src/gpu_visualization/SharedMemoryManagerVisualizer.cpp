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
#include <gpu_visualization/SharedMemoryManagerVisualizer.h>
#include <gpu_visualization/SharedMemoryManager.h>

using namespace boost::interprocess;
namespace gpu_voxels {
namespace visualization {

SharedMemoryManagerVisualizer::SharedMemoryManagerVisualizer()
{
  shmm = new SharedMemoryManager(shm_segment_name_visualizer, true);
}

SharedMemoryManagerVisualizer::~SharedMemoryManagerVisualizer()
{
  delete shmm;
}

bool SharedMemoryManagerVisualizer::getCameraTargetPoint(glm::vec3& target)
{
  std::pair<Vector3f*, std::size_t> res_s = shmm->getMemSegment().find<Vector3f>(shm_variable_name_target_point.c_str());
  if (res_s.second != 0)
  {
    Vector3f t = *(res_s.first);
    target = glm::vec3(t.x, t.y, t.z);
    return true;
  }
  return false;
}

DrawTypes SharedMemoryManagerVisualizer::getDrawTypes()
{
  std::pair<DrawTypes*, std::size_t> res_s = shmm->getMemSegment().find<DrawTypes>(shm_variable_name_set_draw_types.c_str());
  if (res_s.second != 0)
  {
    DrawTypes tmp = *(res_s.first);
    //*(res_s.first) = DrawTypes();
    return tmp;
  }
  return DrawTypes();
}

} //end of namespace visualization
} //end of namespace gpu_voxels
