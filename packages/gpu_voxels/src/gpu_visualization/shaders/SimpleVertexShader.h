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
 * \date    2013-12-5
 *
 *  \brief The vertexshader without lighting.
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_VISUALIZATION_SHADERS_SIMPLE_VERTEX_SHADER_H_INCLUDED
#define GPU_VOXELS_VISUALIZATION_SHADERS_SIMPLE_VERTEX_SHADER_H_INCLUDED

namespace gpu_voxels {
namespace visualization {

class SimpleVertexShader
{
public:

  static const char* get()
  {
    static std::string foo(
                           "#version 410 core\n"
                           "\n"
                           "// Input vertex data, different for all executions of this shader.\n"
                           "layout(location = 0) in vec3 vertexPosition_modelspace;\n"
                           "layout(location = 2) in vec4 vtranslation_and_scale; //(x,y,z) contain the translation and (w) the scal factor\n"
                           "\n"
                           "// Ouput data\n"
                           "out vec4 fragmentColor;\n"
                           "\n"
                           "// Values that stay constant for the whole mesh.\n"
                           "uniform mat4 VP;\n"
                           "\n"
                           "uniform vec4 startColor;\n"
                           "uniform vec4 endColor;\n"
                           "// if interpolation is false, no color interpolation takes place. The startColor will be used.\n"
                           "uniform bool interpolation;\n"
                           "uniform vec3 interpolationLength;\n"
                           "uniform vec3 translationOffset;\n"
                           "\n"
                           "void main(){\n"
                           "	vec3 v_translation = vtranslation_and_scale.xyz + translationOffset;\n"
                           "	float scale = vtranslation_and_scale.w;\n"
                           "\n"
                           "	// this matrix scales the vertex and then translate it\n"
                           "	mat4 M = mat4(vec4(scale,0,0,0),\n"
                           "				  vec4(0,scale,0,0),\n"
                           "				  vec4(0,0,scale,0),\n"
                           "				  vec4(v_translation,1));\n"
                           "\n"
                           "// calculate position\n"
                           "   gl_Position = VP * M * vec4(vertexPosition_modelspace, 1.0f);\n"
                           "   gl_PointSize = 6.f;\n"
                           "\n"
                           "// calculate the interpolated color\n"
                           "   if (interpolation) {\n"
                           "	  float vertexPos_world_z = v_translation.z + scale * vertexPosition_modelspace.z;\n"
                           "   	  float a = (mod(vertexPos_world_z, interpolationLength.z)) / interpolationLength.z;\n"
                           "   	  a = a > 0.5f ? -2*a+2 : 2*a;\n"
                           "	  fragmentColor = (a * startColor + (1 - a) * endColor);\n"
                           "   }\n"
                           "   else {\n"
                           "      fragmentColor = startColor;\n"
                           "   }\n"
                           " }\n"
      );
        return foo.c_str();
  }
};

} // end of ns
} // end of ns

#endif
