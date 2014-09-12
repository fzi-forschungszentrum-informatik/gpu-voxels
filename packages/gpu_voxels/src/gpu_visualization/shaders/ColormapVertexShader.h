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
 * \date    2013-12-2
 *
 *  \brief The vertexshader for the colormap.
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_VISUALIZATION_SHADERS_COLORMAP_VERTEX_SHADER_H_INCLUDED
#define GPU_VOXELS_VISUALIZATION_SHADERS_COLORMAP_VERTEX_SHADER_H_INCLUDED

namespace gpu_voxels {
namespace visualization {

class ColormapVertexShader
{
public:

  static const char* get()
  {
    static std::string foo(
                           "#version 410 core\n"
                           "\n"
                           "// Input vertex data, different for all executions of this shader.\n"
                           "layout(location = 0) in vec3 vertexPosition_modelspace;\n"
                           "layout(location = 2) in vec4 vtranslation_and_scale;\n"
                           "\n"
                           "// Ouput data\n"
                           "out vec3 fragmentColor;\n"
                           "\n"
                           "// Values that stay constant for the whole mesh.\n"
                           "uniform mat4 VP;\n"
                           "uniform int axis; // 0=x-axis, 1=y, 2=z\n"
                           "\n"
                           "void main() {\n"
                           "    vec3 v_translation = vtranslation_and_scale.xyz;\n"
                           "    float scale = vtranslation_and_scale.w;\n"
                           "\n"
                           "// this matrix scales the vertex and then translate it\n"
                           "    mat4 M = mat4(vec4(scale,0,0,0),\n"
                           "                  vec4(0,scale,0,0),\n"
                           "                  vec4(0,0,scale,0),\n"
                           "                  vec4(v_translation,1));\n"
                           "\n"
                           "// calculate position\n"
                           "   gl_Position = VP * M * vec4(vertexPosition_modelspace, 1.0f);\n"
                           "\n"
                           "   float vertexPos_world_axis = v_translation[axis] + scale * vertexPosition_modelspace[axis];\n"
                           "   int v = int(vertexPos_world_axis);\n"
                           "   int r = (v / (256 * 256)) % 256;\n"
                           "   int g = (v / 256) % 256;\n"
                           "   int b = v % 256;\n"
                           "\n"
                           "   fragmentColor = vec3(r/255.f, g/255.f, b/255.f);\n"
                           "   if(v < 0) {\n"
                           "    fragmentColor = vec3(1.f);\n"
                           "   }\n"
                           "}\n"
                           );

    return foo.c_str();
  }
};

} // end of ns
} // end of ns

#endif



 
