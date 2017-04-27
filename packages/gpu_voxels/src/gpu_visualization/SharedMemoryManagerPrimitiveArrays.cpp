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
 */
//----------------------------------------------------------------------
#include <gpu_visualization/SharedMemoryManagerPrimitiveArrays.h>
#include <gpu_visualization/SharedMemoryManager.h>

using namespace boost::interprocess;
namespace gpu_voxels {
namespace visualization {


SharedMemoryManagerPrimitiveArrays::SharedMemoryManagerPrimitiveArrays()
{
  shmm = new SharedMemoryManager(shm_segment_name_primitive_array, true);
}

SharedMemoryManagerPrimitiveArrays::~SharedMemoryManagerPrimitiveArrays()
{
  delete shmm;
}

uint32_t SharedMemoryManagerPrimitiveArrays::getNumberOfPrimitiveArraysToDraw()
{
  std::pair<uint32_t*, std::size_t> res = shmm->getMemSegment().find<uint32_t>(
      shm_variable_name_number_of_primitive_arrays.c_str());
  if (res.second == 0)
  {
    return 0;
  }

  return *res.first;
}

std::string SharedMemoryManagerPrimitiveArrays::getNameOfPrimitiveArray(const uint32_t index)
{
  std::string prim_array_name;
  std::string index_str = boost::lexical_cast<std::string>(index);
  std::string prim_array_name_var_name = shm_variable_name_primitive_array_name + index_str;
  std::pair<char*, std::size_t> res_n = shmm->getMemSegment().find<char>(prim_array_name_var_name.c_str());
  if (res_n.second == 0)
  { /*If the segment couldn't be find or the string is empty*/
    prim_array_name = "primitive_array_" + index_str;
    return prim_array_name;
  }
  prim_array_name.assign(res_n.first, res_n.second);
  return prim_array_name;
}


bool SharedMemoryManagerPrimitiveArrays::getPrimitivePositions(const uint32_t index, Vector4f **d_positions, uint32_t& size,
                                                          primitive_array::PrimitiveType& type)
{
  bool error = false;
  std::string index_str = boost::lexical_cast<std::string>(index);
  std::string handler_name = shm_variable_name_primitive_array_handler_dev_pointer + index_str;
  std::string number_primitives = shm_variable_name_primitive_array_number_of_primitives + index_str;
  std::string type_primitives = shm_variable_name_primitive_array_type + index_str;

  //Find the handler object
  std::pair<cudaIpcMemHandle_t*, std::size_t> res_h = shmm->getMemSegment().find<cudaIpcMemHandle_t>(
      handler_name.c_str());
  error = res_h.second == 0;
  if (!error)
  {
    cudaIpcMemHandle_t handler = *res_h.first;
    // get to device data pointer from the handler
    cudaError_t cuda_error = cudaIpcOpenMemHandle((void**) d_positions, (cudaIpcMemHandle_t) handler,
                                                  cudaIpcMemLazyEnablePeerAccess);
    // the handle is closed by Visualizer.cu
    if (cuda_error == cudaSuccess)
    {
      //Find the number of primitives
      std::pair<uint32_t*, std::size_t> res_d = shmm->getMemSegment().find<uint32_t>(number_primitives.c_str());
      std::pair<primitive_array::PrimitiveType*, std::size_t> res_p = shmm->getMemSegment().find<primitive_array::PrimitiveType>(type_primitives.c_str());
      error = (res_d.second == 0) | (res_p.second == 0);
      if (!error)
      {
        size = *res_d.first;
        type = *res_p.first;
        LOGGING_DEBUG_C(SharedMemManager, SharedMemoryManagerPrimitiveArrays, "Number of primitives in array: " << size << " Type: " << type << endl);
        return true;
      }else{
        LOGGING_ERROR_C(SharedMemManager, SharedMemoryManagerPrimitiveArrays, "Primitive Arrays count or type could not be read from SHM." << endl);
      }
    }else{
      LOGGING_ERROR_C(SharedMemManager, SharedMemoryManagerPrimitiveArrays, "Primitive Arrays Handler could not be opened! Error was " << cuda_error << endl);
    }
  }else{
    LOGGING_ERROR_C(SharedMemManager, SharedMemoryManagerPrimitiveArrays, "Primitive Arrays Handler not found!" << endl);
  }
  cudaIpcCloseMemHandle(*d_positions);
  /*If an error occurred */
  size = 0;
  type = primitive_array::ePRIM_INITIAL_VALUE;
  d_positions = NULL;
  return false;
}
bool SharedMemoryManagerPrimitiveArrays::hasPrimitiveBufferChanged(const uint32_t index)
{
  std::string index_str = boost::lexical_cast<std::string>(index);
  std::string swapped_buffer_name = shm_variable_name_primitive_array_data_changed + index_str;
  std::pair<bool*, std::size_t> swapped = shmm->getMemSegment().find<bool>(swapped_buffer_name.c_str());

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
void SharedMemoryManagerPrimitiveArrays::setPrimitiveBufferChangedToFalse(const uint32_t index)
{
  std::string index_str = boost::lexical_cast<std::string>(index);
  std::string swapped_buffer_name = shm_variable_name_primitive_array_data_changed + index_str;
  std::pair<bool*, std::size_t> swapped = shmm->getMemSegment().find<bool>(swapped_buffer_name.c_str());

  if (swapped.second)
  {
    *swapped.first = false;
  }
}

} //end of namespace visualization
} //end of namespace gpu_voxels
