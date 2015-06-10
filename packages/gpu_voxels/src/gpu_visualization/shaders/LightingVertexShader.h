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
 * \date    2014-2-15
 *
 *  \brief The vertexshader with lighting.
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_VISUALIZATION_SHADERS_LIGHTING_VERTEX_SHADER_H_INCLUDED
#define GPU_VOXELS_VISUALIZATION_SHADERS_LIGHTING_VERTEX_SHADER_H_INCLUDED


namespace gpu_voxels {
namespace visualization {

class LightingVertexShader
{
public:

  static const char* get()
  {
    static std::string foo(
                           "#version 410 core\n"
                           "\n"
                           "// Input vertex data, different for all executions of this shader.\n"
                           "layout(location = 0) in vec3 vertexPosition_modelspace;\n"
                           "layout(location = 1) in vec3 vertexNormal_modelspace;\n"
                           "layout(location = 2) in vec4 vtranslation_and_scale;//(x,y,z) contain the translation and (w) the scal factor\n"
                           "\n"
                           "// Ouput data\n"
                           "out vec4 fragmentColor;\n"
                           "out vec3 normal_cameraspace;\n"
                           "out vec3 position_cameraspace;\n"
                           "\n"
                           "\n"
                           "// Values that stay constant for the whole mesh.\n"
                           "uniform mat4 VP; // View-Projektion-Matrix\n"
                           "uniform mat4 V;  // View-Matrix\n"
                           "uniform mat4 V_inverse_transpose; // inverse-transpose View-Matrix\n"
                           "\n"
                           "\n"
                           "uniform vec4 startColor;\n"
                           "uniform vec4 endColor;\n"
                           "// if interpolation is false, no color interpolation takes place. The startColor will be used.\n"
                           "uniform bool interpolation;\n"
                           "uniform vec3 interpolationLength;\n"
                           "uniform vec3 translationOffset;\n"
                           "\n"
                           "void main(){\n"
                           "	gl_PointSize = 6.f;\n"
                           "	vec3 v_translation = vtranslation_and_scale.xyz + translationOffset;\n"
                           "	float scale = vtranslation_and_scale.w;\n"
                           "\n"
                           "	mat4 M = mat4(vec4(scale,0,0,0),\n"
                           "		      vec4(0,scale,0,0),\n"
                           "		      vec4(0,0,scale,0),\n"
                           "		      vec4(v_translation,1));\n"
                           "\n"
                           "	mat4 M_inverse_transpose = mat4(1/scale,0,0,-v_translation.x/scale,\n"
                           "	                                0,1/scale,0,-v_translation.y/scale,\n"
                           "	                                0,0,1/scale,-v_translation.z/scale,\n"
                           "	                                0,0,0,                1);\n"
                           "\n"
                           "////////////// apply vertex and normal transformations /////////////////////////////////\n"
                           "	gl_Position = VP * M * vec4(vertexPosition_modelspace, 1.0f);\n"
                           "\n"
                           "	position_cameraspace = vec3(V * M * vec4(vertexPosition_modelspace,1));\n"
                           "	normal_cameraspace = vec3(V_inverse_transpose * M_inverse_transpose * vec4(vertexNormal_modelspace,0));\n"
                           "\n"
                           "///////caluclate the color of the vertex /////////////////////////////////////\n"
                           "   if (interpolation) {\n"
                           "	  float vertexPos_world_z = v_translation.z + vertexPosition_modelspace.z;\n"
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
