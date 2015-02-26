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
 * \author  Florian Drews
 * \date    2014-06-18
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_HELPERS_MANAGEDMAP_H_INCLUDED
#define GPU_VOXELS_HELPERS_MANAGEDMAP_H_INCLUDED

#include <gpu_voxels/GpuVoxels.h>
#include <gpu_voxels/vis_interface/VisProvider.h>

namespace gpu_voxels {

struct ManagedMap
{

  ManagedMap(GpuVoxelsMapSharedPtr map_shared_ptr, VisProviderSharedPtr vis_provider_shared_ptr)
  {
    this->map_shared_ptr = map_shared_ptr;
    this->vis_provider_shared_ptr = vis_provider_shared_ptr;
  }

  GpuVoxelsMapSharedPtr map_shared_ptr;
  VisProviderSharedPtr vis_provider_shared_ptr;
};

} // end of ns

#endif
