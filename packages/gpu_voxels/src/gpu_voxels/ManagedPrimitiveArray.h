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
 * \date    2015-01-05
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_HELPERS_MANAGEDPRIMITIVEARRAY_H_INCLUDED
#define GPU_VOXELS_HELPERS_MANAGEDPRIMITIVEARRAY_H_INCLUDED

#include <gpu_voxels/primitive_array/PrimitiveArray.h>
#include <gpu_voxels/vis_interface/VisProvider.h>

namespace gpu_voxels {

struct ManagedPrimitiveArray
{

  ManagedPrimitiveArray(primitive_array::PrimitiveArraySharedPtr prim_array_shared_ptr, VisProviderSharedPtr vis_provider_shared_ptr)
  {
    this->prim_array_shared_ptr = prim_array_shared_ptr;
    this->vis_provider_shared_ptr = vis_provider_shared_ptr;
  }

  primitive_array::PrimitiveArraySharedPtr prim_array_shared_ptr;
  VisProviderSharedPtr vis_provider_shared_ptr;
};

} // end of ns

#endif
