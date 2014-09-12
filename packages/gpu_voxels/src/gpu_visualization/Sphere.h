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
 * \brief this primitive represents a sphere.
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_VISUALIZATION_SPHERE_H_INCLUDED
#define GPU_VOXELS_VISUALIZATION_SPHERE_H_INCLUDED

#include <gpu_visualization/Primitive.h>

namespace gpu_voxels {
namespace visualization {

class Sphere: public Primitive
{
public:

  Sphere()
    : m_position(0),
      m_radius(10),
      longitudeEntries(16),
      latitudeEntries(16),
      elementbuffer_north_pole(0),
      elementbuffer_south_pole(0),
      elementbuffer_sphere_body(0),
      size_elementbuffer_north_pole(0),
      size_elementbuffer_south_pole(0),
      size_elementbuffer_sphere_body(0)
  {
    m_color = glm::vec4(1.f);
    m_is_created = false;
  }

  Sphere(glm::vec4 color, glm::vec3 position, float radius, uint32_t resolution)
    : longitudeEntries(resolution),
      latitudeEntries(resolution),
      elementbuffer_north_pole(0),
      elementbuffer_south_pole(0),
      elementbuffer_sphere_body(0),
      size_elementbuffer_north_pole(0),
      size_elementbuffer_south_pole(0),
      size_elementbuffer_sphere_body(0)
  {
    m_color = color;
    m_position = position;
    m_radius = radius;
    m_is_created = false;
  }

  ~Sphere()
  {
    glDeleteBuffers(1, &m_vbo);
    glDeleteBuffers(1, &elementbuffer_north_pole);
    glDeleteBuffers(1, &elementbuffer_south_pole);
    glDeleteBuffers(1, &elementbuffer_sphere_body);
  }

  /**
   * Creates all necessary VBOs for this sphere.
   */
  virtual void create(bool with_lighting)
  {
    std::vector<uint32_t> indices_north_pole;
    std::vector<uint32_t> indices_south_pole;
    std::vector<uint32_t> indices_body;

    std::vector<glm::vec3> vertices;
    std::vector<uint32_t> indices;

    for (float j = 1; j < latitudeEntries + 1; j++)
    {
      for (float i = 0; i < longitudeEntries; i++)
      {
        float latitude = j / (latitudeEntries + 1) * M_PI - M_PI_2; // ]-PI/2, PI/2[ without the edges, so that the poles get excluded
        float longitude = i / (longitudeEntries) * 2 * M_PI; // [0, 2*PI]
        float x = m_radius * cosf(latitude) * cosf(longitude) + m_position.x;
        float z = m_radius * cosf(latitude) * sinf(longitude) + m_position.z;
        float y = m_radius * sinf(latitude) + m_position.y;
        vertices.push_back(glm::vec3(x, y, z));
      }
    }
    // assert(vertices.size() == size_t(longitudeEntries * latitudeEntries));
    //north pole of the sphere
    vertices.push_back(m_position + glm::vec3(0, m_radius, 0));

    //south pole of the sphere
    vertices.push_back(m_position - glm::vec3(0, m_radius, 0));

    ///////////////////////////////insert indices of north pole
    indices_north_pole.push_back(vertices.size() - 2); // north pole is at index (size - 2)
    for (int32_t i = longitudeEntries - 1; i >= 0; i--)
    {
      uint32_t index = (latitudeEntries - 1) * longitudeEntries + i;
      indices_north_pole.push_back(index);
    }
    //to close the fan add the last point from points again
    indices_north_pole.push_back((latitudeEntries - 1) * longitudeEntries + longitudeEntries - 1);

    ////////////////////////////////////insert indices of south pole
    indices_south_pole.push_back(vertices.size() - 1); // south pole is at index (size - 1)
    for (uint32_t i = 0; i < longitudeEntries; i++)
    {
      indices_south_pole.push_back(i);
    }
    // to close the fan add the first point from points again
    indices_south_pole.push_back(0);

    //////////////////////////////// insert indices of sphere body
    uint32_t index = 0;
    uint32_t next_index;
    for (uint32_t latitude = 1; latitude < latitudeEntries; latitude++)
    {
      next_index = longitudeEntries * latitude;
      for (uint32_t longitude = 0; longitude < longitudeEntries; longitude++)
      {
        indices_body.push_back(index + longitude);
        indices_body.push_back(next_index + longitude);
      }
      // repeat the first one to close the strip
      uint32_t longitude = 0;
      indices_body.push_back(index + longitude);
      indices_body.push_back(next_index + longitude);
      indices_body.push_back(0xffff); //insert restart index
      index = next_index;
    }
    glDeleteBuffers(1, &m_vbo);
    glDeleteBuffers(1, &elementbuffer_north_pole);
    glDeleteBuffers(1, &elementbuffer_south_pole);
    glDeleteBuffers(1, &elementbuffer_sphere_body);

    glGenBuffers(1, &m_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices.front(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glGenBuffers(1, &elementbuffer_north_pole);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer_north_pole);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_north_pole.size() * sizeof(uint32_t),
                 &indices_north_pole[0], GL_STATIC_DRAW);

    glGenBuffers(1, &elementbuffer_south_pole);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer_south_pole);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_south_pole.size() * sizeof(uint32_t),
                 &indices_south_pole[0], GL_STATIC_DRAW);

    glGenBuffers(1, &elementbuffer_sphere_body);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer_sphere_body);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_body.size() * sizeof(uint32_t), &indices_body[0],
                 GL_STATIC_DRAW);

    size_elementbuffer_north_pole = indices_north_pole.size();
    size_elementbuffer_south_pole = indices_south_pole.size();
    size_elementbuffer_sphere_body = indices_body.size();

    m_is_created = true;
    m_lighting_mode = with_lighting;
  }

  /**
   * draw the sphere multiple times.
   * All uniform variables of the shaders must be set before call.
   *
   * @param number_of_draws: the number of draw calls for this sphere.
   */
  virtual void draw(uint32_t number_of_draws, bool with_lighting)
  {
    if (!m_is_created)
    { /*create a new cuboid if it hasn't been created jet
     (no check for lighting necessary because the vbo is for both identically )*/
      create(with_lighting);
    }
    glPrimitiveRestartIndex(0xffff);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

    glEnableVertexAttribArray(0);
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
      // the Normals of a sphere are the normalized positions so simple use the positions again
      glVertexAttribPointer(1, // attribute 0 (must match the layout in the shader).
          3, // size
          GL_FLOAT, // type
          GL_FALSE, // normalized?
          0, // stride
          (void*) 0 // array buffer offset
          );
    }
    else
    {
      glDisableVertexAttribArray(1);
    }

    ExitOnGLError("ERROR: Couldn't set the vertex attribute pointer.");
    ///////////////////////////draw north pole////////////////////////////
    // Index buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer_north_pole);

    // Draw the triangles !
    glDrawElementsInstanced(GL_TRIANGLE_FAN, size_elementbuffer_north_pole, GL_UNSIGNED_INT, 0,
                            number_of_draws);

//    glDrawElements(GL_TRIANGLE_FAN, // mode
//        size_elementbuffer_north_pole, // count
//        GL_UNSIGNED_INT, // type
//        (void*) 0 // element array buffer offset
//        );

    ///////////////////////////draw south pole////////////////////////////
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer_south_pole);

    // Draw the triangles !
    glDrawElementsInstanced(GL_TRIANGLE_FAN, // mode
        size_elementbuffer_south_pole, // count
        GL_UNSIGNED_INT, // type
        (void*) 0 // element array buffer offset
        , number_of_draws);

    ///////////////////////////draw sphere body////////////////////////////
    glEnable(GL_PRIMITIVE_RESTART);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer_sphere_body);

    // Draw the triangles !
    glDrawElementsInstanced(GL_TRIANGLE_STRIP, // mode
        size_elementbuffer_sphere_body, // count
        GL_UNSIGNED_INT, // type
        (void*) 0 // element array buffer offset
        , number_of_draws);
    glDisable(GL_PRIMITIVE_RESTART);
  }

private:

  glm::vec3 m_position;
  float m_radius;

  uint32_t longitudeEntries;
  uint32_t latitudeEntries;

  GLuint elementbuffer_north_pole;
  GLuint elementbuffer_south_pole;
  GLuint elementbuffer_sphere_body;

  uint32_t size_elementbuffer_north_pole;
  uint32_t size_elementbuffer_south_pole;
  uint32_t size_elementbuffer_sphere_body;

};

} // end of namespace visualization
} // end of namespace gpu_voxels

#endif
