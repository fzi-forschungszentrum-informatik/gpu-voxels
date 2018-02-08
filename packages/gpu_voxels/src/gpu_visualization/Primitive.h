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
 * \date    2014-02-24
 *
 * \brief   A base frame for different primitives.
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_VISUALIZATION_PRIMITIVE_H_INCLUDED
#define GPU_VOXELS_VISUALIZATION_PRIMITIVE_H_INCLUDED

#define GLM_FORCE_RADIANS
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <gpu_visualization/Utils.h>
#include <gpu_visualization/visualizerDefines.h>

namespace gpu_voxels {
namespace visualization {

class Primitive
{
protected:

  GLuint m_vbo;
  glm::vec4 m_color;
  bool m_is_created;
  bool m_lighting_mode;

  Primitive()
    : m_vbo(0),
      m_color(glm::vec4(1.f)),
      m_is_created(false),
      m_lighting_mode(false)
  {
  }

public:

  virtual ~Primitive()
  {
  }
  virtual void draw(uint32_t number_of_draws, bool with_lighting)
  {
  }

  virtual void create(bool with_lighting)
  {
  }

  glm::vec4 getColor()
  {
    return m_color;
  }
};

} // end of namespace visualization
} // end of namespace gpu_voxels

#endif
