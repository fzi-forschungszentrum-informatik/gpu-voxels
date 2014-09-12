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
using namespace boost::interprocess;
namespace gpu_voxels {
namespace visualization {

bool SharedMemoryManagerVisualizer::getCameraTargetPoint(glm::vec3& target)
{
  std::pair<Vector3f*, std::size_t> res_s = m_segment.find<Vector3f>(shm_variable_name_target_point.c_str());
  if (res_s.second != 0)
  {
    Vector3f t = *(res_s.first);
    target = glm::vec3(t.x, t.y, t.z);
    return true;
  }
  return false;
}
bool SharedMemoryManagerVisualizer::getPrimitivePositions(glm::vec3*& d_positions, uint32_t& size,
                                                          PrimitiveTypes& type)
{
  bool error = false;
  std::string handler_name = shm_variable_name_primitive_handler_dev_pointer;
  std::string number_primitives = shm_variable_name_number_of_primitives;

  //Find the handler object
  std::pair<cudaIpcMemHandle_t*, std::size_t> res_h = m_segment.find<cudaIpcMemHandle_t>(
      handler_name.c_str());
  error = res_h.second == 0;
  glm::vec3* dev_data_pointer;

  if (!error)
  {
    cudaIpcMemHandle_t handler = *res_h.first;
    // get to device data pointer from the handler
    cudaError_t cuda_error = cudaIpcOpenMemHandle((void**) &dev_data_pointer, (cudaIpcMemHandle_t) handler,
                                                  cudaIpcMemLazyEnablePeerAccess);
    if (cuda_error == cudaSuccess)
    {
      //Find the number of cubes
      std::pair<uint32_t*, std::size_t> res_d = m_segment.find<uint32_t>(number_primitives.c_str());
      std::string type_primitives = shm_variable_name_primitive_type;
      std::pair<PrimitiveTypes*, std::size_t> res_p = m_segment.find<PrimitiveTypes>(type_primitives.c_str());
      error = (res_d.second == 0) | (res_p.second == 0);
      if (!error)
      {
        d_positions = dev_data_pointer;
        size = *res_d.first;
        type = *res_p.first;
        return true;
      }
    }
  }
  cudaIpcCloseMemHandle(dev_data_pointer);
  /*If an error occurred */
  return false;
}
bool SharedMemoryManagerVisualizer::hasPrimitiveBufferChanged()
{
  std::string swapped_buffer_name = shm_variable_name_primitive_buffer_changed;
  std::pair<bool*, std::size_t> swapped = m_segment.find<bool>(swapped_buffer_name.c_str());

  if (swapped.second != 0)
  {
    return *swapped.first;
  }
  else
  {
    return false;
  }
}
/**
 * Sets the shared memory variable buffer_changed_primitive to false.
 */
void SharedMemoryManagerVisualizer::setPrimitiveBufferChangedToFalse()
{
  std::string swapped_buffer_name = shm_variable_name_primitive_buffer_changed;
  std::pair<bool*, std::size_t> swapped = m_segment.find<bool>(swapped_buffer_name.c_str());

  if (swapped.second)
  {
    *swapped.first = false;
  }
}

DrawTypes SharedMemoryManagerVisualizer::getDrawTypes()
{
  std::pair<DrawTypes*, std::size_t> res_s = m_segment.find<DrawTypes>(shm_variable_name_set_draw_types.c_str());
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
