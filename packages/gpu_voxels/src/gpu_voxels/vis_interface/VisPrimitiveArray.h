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
#ifndef GPU_VOXELS_VISPRIMITIVEARRAY_H_INCLUDED
#define GPU_VOXELS_VISPRIMITIVEARRAY_H_INCLUDED

#include <gpu_voxels/vis_interface/VisProvider.h>
#include <gpu_voxels/vis_interface/VisualizerInterface.h>
#include <gpu_voxels/primitive_array/PrimitiveArray.h>

#include <cuda_runtime.h>

namespace gpu_voxels {

class VisPrimitiveArray: public VisProvider
{
public:

  VisPrimitiveArray(primitive_array::PrimitiveArray* primitive_array, std::string array_name);

  virtual ~VisPrimitiveArray();

  virtual bool visualize(const bool force_repaint = true);

  virtual uint32_t getResolutionLevel() { return 0; }

protected:
  primitive_array::PrimitiveArray* m_primitive_array;
  cudaIpcMemHandle_t* m_shm_memHandle;
  float* m_shm_primitive_diameter;
  uint32_t* m_shm_num_primitives;
  primitive_array::PrimitiveType* m_shm_primitive_type;
  bool* m_shm_primitive_array_changed;
};

}

#endif
