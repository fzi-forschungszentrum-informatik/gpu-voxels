// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is from: http://www.opengl-tutorial.org/
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Matthias Wagner
 * \date    2013-12-17
 *
 *  \brief Loads and compiles the fragment and vertex shader.
 *              from: http://www.opengl-tutorial.org/
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_VISUALIZATION_SHADER_H_INCLUDED
#define GPU_VOXELS_VISUALIZATION_SHADER_H_INCLUDED

// 1) adjust CSG
namespace gpu_voxels {
namespace visualization {

GLuint loadShaders(const char *vertex_shader, const char *fragment_shader);

} // end of namespace visualization
} // end of namespace gpu_voxels
#endif
