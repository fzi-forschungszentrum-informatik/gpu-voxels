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
 *  \brief The fragmentshader with lighting.
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_VISUALIZATION_SHADERS_LIGHTING_FRAGMENT_SHADER_H_INCLUDED
#define GPU_VOXELS_VISUALIZATION_SHADERS_LIGHTING_FRAGMENT_SHADER_H_INCLUDED

namespace gpu_voxels {
namespace visualization {

class LightingFragmentShader
{
public:

  static const char* get()
  {
    static std::string foo(
                           "#version 410 core\n"
                           "\n"
                           "// Ouput data\n"
                           "out vec4 color_out;\n"
                           "\n"
                           "// Input data\n"
                           "in vec4 fragmentColor;\n"
                           "\n"
                           "\n"
                           "in vec3 normal_cameraspace;\n"
                           "in vec3 position_cameraspace;\n"
                           "\n"
                           "\n"
                           "uniform mat4 V;\n"
                           "uniform vec3 lightPosition_worldspace;\n"
                           "uniform vec3 lightIntensity;\n"
                           "\n"
                           "void main()\n"
                           "{// implement Phong-Lighting\n"
                           "   vec3 normal_cameraspace_normalized = normalize(normal_cameraspace);\n"
                           "   if(dot(position_cameraspace,normal_cameraspace_normalized) > 0.f) {\n"
                           "   	// if the fragment is not visible discard it\n"
                           "        discard;\n"
                           "    }\n"
                           "    float n = 50; // Phong exponent\n"
                           "	vec3 kd = fragmentColor.rgb; // diffuse color\n"
                           "\n"
                           "    vec3 lightPos_cameraspace = (V * vec4(lightPosition_worldspace,1)).xyz;\n"
                           "    float distance = length(lightPos_cameraspace - position_cameraspace);\n"
                           "    float attenuation = distance > 1.f ? 1.f / pow(distance, 2.f) : 1.f;\n"
                           "\n"
                           "    // the attenuated light intensity\n"
                           "    vec3 lightIntensityAttenuation = min(lightIntensity * attenuation, 1.f);\n"
                           "\n"
                           "    vec3 light = normalize(lightPos_cameraspace - position_cameraspace);\n"
                           "\n"
                           "	vec3 ambient_color = fragmentColor.rgb * vec3(0.4f);\n"
                           "	vec3 diffus_color = vec3(0.f);\n"
                           "\n"
                           "	float NL = max(0.f, dot(normalize(normal_cameraspace_normalized),light));\n"
                           "    if(NL > 0.f) {\n"
                           "       	diffus_color = kd * lightIntensityAttenuation *  NL;\n"
                           "   }\n"
                           "\n"
                           "   color_out = vec4(ambient_color + diffus_color, fragmentColor.a);\n"
                           "}\n"
                           "\n"
                           );
        return foo.c_str();
    }
};
} // end of ns
} // end of ns

#endif
