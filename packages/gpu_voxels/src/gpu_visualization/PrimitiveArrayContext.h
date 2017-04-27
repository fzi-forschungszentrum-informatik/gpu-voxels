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
 * \date    2015-01-07
 *
 *\brief   Saves all necessary stuff to draw a primitive array.
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_VISUALIZATION_PRIMITIVE_ARRAY_CONTEXT_H_INCLUDED
#define GPU_VOXELS_VISUALIZATION_PRIMITIVE_ARRAY_CONTEXT_H_INCLUDED

#include <vector_types.h>
#include <stdlib.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <gpu_voxels/primitive_array/PrimitiveArray.h>
#include <gpu_visualization/logging/logging_visualization.h>

#include "Cuboid.h"
#include "Sphere.h"
#include "DataContext.h"


namespace gpu_voxels {
namespace visualization {

typedef std::pair<glm::vec4, glm::vec4> colorPair;

class PrimitiveArrayContext: public DataContext
{
public:
  PrimitiveArrayContext() :
  m_prim_type(primitive_array::ePRIM_INITIAL_VALUE)
  {
  }

  /**
   * Create a default context for the given voxel map.
   */
  PrimitiveArrayContext(std::string prim_array_name) :
  m_prim_type(primitive_array::ePRIM_INITIAL_VALUE)
  {
    m_map_name = prim_array_name;

    m_default_prim = new Cuboid(glm::vec4(0.f, 0.f, 0.f, 1.f),
                                glm::vec3(0.f, 0.f, 0.f),
                                glm::vec3(1.f, 1.f, 1.f));
  }

  virtual void updateVBOOffsets()
  {
    LOGGING_ERROR_C(Context, PrimitiveArrayContext, "updateVBOOffsets Function NOT IMPLEMENTED!" << endl);
  }

  virtual void updateCudaLaunchVariables(Vector3ui supervoxel_size = Vector3ui(1))
  {
    LOGGING_ERROR_C(Context, PrimitiveArrayContext, "updateCudaLaunchVariables Function NOT IMPLEMENTED!" << endl);
  }

  primitive_array::PrimitiveType m_prim_type;

};

}  // end of ns
}  // end of ns

#endif
