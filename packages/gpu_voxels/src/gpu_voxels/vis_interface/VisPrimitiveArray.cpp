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
 * \author  Andreas Hermann
 * \date    2014-12-15
 *
 */
//----------------------------------------------------------------------/*
#include <gpu_voxels/vis_interface/VisPrimitiveArray.h>
#include <gpu_voxels/helpers/cuda_handling.h>
#include <cstdio>

std::ostream& operator << (std::ostream& os, const cudaIpcMemHandle_t& handle)
{
  os << int(handle.reserved[0]);
  for (int i=1; i<CUDA_IPC_HANDLE_SIZE; ++i)
  {
    os << " " << int(handle.reserved[i]);
  }
  return os;
}

namespace gpu_voxels {

VisPrimitiveArray::VisPrimitiveArray(primitive_array::PrimitiveArray* primitive_array, std::string array_name) :
    VisProvider(shm_segment_name_primitive_array, array_name), /**/
    m_primitive_array(primitive_array), /**/
    m_shm_memHandle(NULL), /**/
    m_shm_primitive_diameter(NULL), /**/
    m_shm_num_primitives(NULL), /**/
    m_shm_primitive_type(NULL), /**/
    m_shm_primitive_array_changed(NULL)
{
}

VisPrimitiveArray::~VisPrimitiveArray()
{
}

bool VisPrimitiveArray::visualize(const bool force_repaint)
{
  if (force_repaint)
  {
    openOrCreateSegment();
    uint32_t shared_mem_id;
    if (m_shm_memHandle == NULL)
    {
      // there should only be one segment of number_of_primitive_arrays
      std::pair<uint32_t*, std::size_t> r = m_segment.find<uint32_t>(
          shm_variable_name_number_of_primitive_arrays.c_str());
      if (r.second == 0)
      { // if it doesn't exists ..
        m_segment.construct<uint32_t>(shm_variable_name_number_of_primitive_arrays.c_str())(1);
        shared_mem_id = 0;
      }
      else
      { // if it exists increase it by one
        shared_mem_id = *r.first;
        (*r.first)++;
      }
      // get shared memory pointer
      std::stringstream id;
      id << shared_mem_id;
      m_shm_memHandle = m_segment.find_or_construct<cudaIpcMemHandle_t>(
          std::string(shm_variable_name_primitive_array_handler_dev_pointer + id.str()).c_str())(
            cudaIpcMemHandle_t());
      m_shm_primitive_diameter = m_segment.find_or_construct<float>(
          std::string(shm_variable_name_primitive_array_prim_diameter + id.str()).c_str())(0.0f);
      m_shm_mapName = m_segment.find_or_construct_it<char>(
          std::string(shm_variable_name_primitive_array_name + id.str()).c_str())[m_map_name.size()](
          m_map_name.data());
      m_shm_primitive_type = m_segment.find_or_construct<primitive_array::PrimitiveType>(
          std::string(shm_variable_name_primitive_array_type + id.str()).c_str())(m_primitive_array->getPrimitiveType());
      m_shm_num_primitives = m_segment.find_or_construct<uint32_t>(
          std::string(shm_variable_name_primitive_array_number_of_primitives + id.str()).c_str())(m_primitive_array->getNumPrimitives());
      m_shm_primitive_array_changed = m_segment.find_or_construct<bool>(
          std::string(shm_variable_name_primitive_array_data_changed + id.str()).c_str())(true);
    }
    // first open or create and then set the values
    // but only, if data is available
    if(m_primitive_array->getVoidDeviceDataPtr())
    {
      HANDLE_CUDA_ERROR(cudaIpcGetMemHandle(m_shm_memHandle, m_primitive_array->getVoidDeviceDataPtr()));
      *m_shm_primitive_diameter = m_primitive_array->getDiameter();
      *m_shm_primitive_type = m_primitive_array->getPrimitiveType();
      *m_shm_num_primitives = m_primitive_array->getNumPrimitives();
      *m_shm_primitive_array_changed = true;
    }else{
      *m_shm_primitive_array_changed = false;
    }

//    // wait till data was read by visualizer. Otherwise a
//    while(*m_shm_voxelmap_changed)
//      usleep(10000); // sleep 10 ms

    return true;
  }
  return false;
}

} // end of ns
