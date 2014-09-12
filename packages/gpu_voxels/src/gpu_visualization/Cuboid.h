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
 * \brief this primitive represents a cuboid.
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_VISUALIZATION_CUBOID_H_INCLUDED
#define GPU_VOXELS_VISUALIZATION_CUBOID_H_INCLUDED
#include <gpu_visualization/Primitive.h>

namespace gpu_voxels {
namespace visualization {

class Cuboid: public Primitive
{
public:

  Cuboid()
  {
    m_color = glm::vec4(1.f);
    m_position = glm::vec3(0.f);
    m_side_length = glm::vec3(1.f);
    m_is_created = false;
    m_lighting_mode = false;
  }

  Cuboid(glm::vec4 color, glm::vec3 position, glm::vec3 side_length)
  {
    m_position = position;
    m_side_length = side_length;
    m_color = color;
    m_is_created = false;
    m_lighting_mode = false;
  }

  ~Cuboid()
  {
    glDeleteBuffers(1, &m_vbo);
  }

  /**
   * Creates all necessary VBOs for this cuboid.
   */
  virtual void create(bool with_lighting)
  {

    glm::vec3 n_neg_x = glm::vec3(-1, 0, 0);
    glm::vec3 n_pos_x = glm::vec3(1, 0, 0);

    glm::vec3 n_neg_y = glm::vec3(0, -1, 0);
    glm::vec3 n_pos_y = glm::vec3(0, 1, 0);

    glm::vec3 n_neg_z = glm::vec3(0, 0, -1);
    glm::vec3 n_pos_z = glm::vec3(0, 0, 1);

    float x = m_position.x;
    float y = m_position.y;
    float z = m_position.z;
    float _x = x + m_side_length.x - 0.00001f; // to correctly interpolate colors
    float _y = y + m_side_length.y - 0.00001f;
    float _z = z + m_side_length.z - 0.00001f;
    std::vector<glm::vec3> vertices;
    // View from negative x
    vertices.push_back(glm::vec3(x, y, z));
    vertices.push_back(glm::vec3(x, y, _z));
    vertices.push_back(glm::vec3(x, _y, z));
    if (with_lighting)
    {
      vertices.push_back(n_neg_x);
      vertices.push_back(n_neg_x);
      vertices.push_back(n_neg_x);
    }
    vertices.push_back(glm::vec3(x, _y, z));
    vertices.push_back(glm::vec3(x, y, _z));
    vertices.push_back(glm::vec3(x, _y, _z));
    if (with_lighting)
    {
      vertices.push_back(n_neg_x);
      vertices.push_back(n_neg_x);
      vertices.push_back(n_neg_x);
    }
    // View from positive x
    vertices.push_back(glm::vec3(_x, y, z));
    vertices.push_back(glm::vec3(_x, _y, z));
    vertices.push_back(glm::vec3(_x, y, _z));
    if (with_lighting)
    {
      vertices.push_back(n_pos_x);
      vertices.push_back(n_pos_x);
      vertices.push_back(n_pos_x);
    }
    vertices.push_back(glm::vec3(_x, _y, z));
    vertices.push_back(glm::vec3(_x, _y, _z));
    vertices.push_back(glm::vec3(_x, y, _z));
    if (with_lighting)
    {
      vertices.push_back(n_pos_x);
      vertices.push_back(n_pos_x);
      vertices.push_back(n_pos_x);
    }
    // View from negative y
    vertices.push_back(glm::vec3(x, y, z));
    vertices.push_back(glm::vec3(_x, y, z));
    vertices.push_back(glm::vec3(x, y, _z));
    if (with_lighting)
    {
      vertices.push_back(n_neg_y);
      vertices.push_back(n_neg_y);
      vertices.push_back(n_neg_y);
    }
    vertices.push_back(glm::vec3(_x, y, z));
    vertices.push_back(glm::vec3(_x, y, _z));
    vertices.push_back(glm::vec3(x, y, _z));
    if (with_lighting)
    {
      vertices.push_back(n_neg_y);
      vertices.push_back(n_neg_y);
      vertices.push_back(n_neg_y);
    }
    // View from positive y
    vertices.push_back(glm::vec3(x, _y, z));
    vertices.push_back(glm::vec3(x, _y, _z));
    vertices.push_back(glm::vec3(_x, _y, z));
    if (with_lighting)
    {
      vertices.push_back(n_pos_y);
      vertices.push_back(n_pos_y);
      vertices.push_back(n_pos_y);
    }
    vertices.push_back(glm::vec3(_x, _y, z));
    vertices.push_back(glm::vec3(x, _y, _z));
    vertices.push_back(glm::vec3(_x, _y, _z));
    if (with_lighting)
    {
      vertices.push_back(n_pos_y);
      vertices.push_back(n_pos_y);
      vertices.push_back(n_pos_y);
    }
    // View from negative z
    vertices.push_back(glm::vec3(x, y, z));
    vertices.push_back(glm::vec3(x, _y, z));
    vertices.push_back(glm::vec3(_x, y, z));
    if (with_lighting)
    {
      vertices.push_back(n_neg_z);
      vertices.push_back(n_neg_z);
      vertices.push_back(n_neg_z);
    }
    vertices.push_back(glm::vec3(x, _y, z));
    vertices.push_back(glm::vec3(_x, _y, z));
    vertices.push_back(glm::vec3(_x, y, z));
    if (with_lighting)
    {
      vertices.push_back(n_neg_z);
      vertices.push_back(n_neg_z);
      vertices.push_back(n_neg_z);
    }
    // View from positive z
    vertices.push_back(glm::vec3(x, y, _z));
    vertices.push_back(glm::vec3(_x, y, _z));
    vertices.push_back(glm::vec3(x, _y, _z));
    if (with_lighting)
    {
      vertices.push_back(n_pos_z);
      vertices.push_back(n_pos_z);
      vertices.push_back(n_pos_z);
    }
    vertices.push_back(glm::vec3(x, _y, _z));
    vertices.push_back(glm::vec3(_x, y, _z));
    vertices.push_back(glm::vec3(_x, _y, _z));
    if (with_lighting)
    {
      vertices.push_back(n_pos_z);
      vertices.push_back(n_pos_z);
      vertices.push_back(n_pos_z);
    }
    glDeleteBuffers(1, &m_vbo);
    glGenBuffers(1, &m_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices.front(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    ExitOnGLError("ERROR: Couldn't create VBOs for the cuboid.");
    m_is_created = true;
    m_lighting_mode = with_lighting;
  }

  /**
   * draw the cuboid multiple times.
   * All uniform variables of the shaders must be set before call.
   *
   * @param number_of_draws: the number of draw calls for this cuboid.
   */
  virtual void draw(uint32_t number_of_draws, bool with_lighting)
  {
    if (!m_is_created || (m_lighting_mode != with_lighting))
    { /*create a new cuboid if it hasn't been created jet or if the lighting mode differs*/
      create(with_lighting);

    }
    uint32_t num_vertices = 36;
    // bind the vbo buffer
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glVertexAttribPointer(0, // attribute 0 (must match the layout in the shader).
        3, // size
        GL_FLOAT, // type
        GL_FALSE, // normalized?
        0, // stride
        (void*) 0 // array buffer offset
        );

    if (with_lighting)
    {
      glEnableVertexAttribArray(1);
      glVertexAttribPointer(1, // attribute 0 (must match the layout in the shader).
          3, // size
          GL_FLOAT, // type
          GL_FALSE, // normalized?
          0, // stride
          (void*) 36 // array buffer offset
          );
      num_vertices *= 2; // for each vertex one normal
    }
    else
    {
      glDisableVertexAttribArray(1);
    }
    ExitOnGLError("ERROR: Couldn't set the vertex attribute pointer.");
    //////////////////////////////////draw the vbo///////////////////////////
    glDrawArraysInstanced(GL_TRIANGLES, 0, num_vertices, number_of_draws);
    ExitOnGLError("ERROR! Couldn't draw the filled triangles.");
  }
  glm::vec3 m_position;
  glm::vec3 m_side_length;
};

} // end of namespace visualization
} // end of namespace gpu_voxels

#endif
