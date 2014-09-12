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
 *  \brief The fragmentshader for the colormap
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_VISUALIZATION_SHADERS_COLORMAP_FRAGMENT_SHADER_H_INCLUDED
#define GPU_VOXELS_VISUALIZATION_SHADERS_COLORMAP_FRAGMENT_SHADER_H_INCLUDED

namespace gpu_voxels {
namespace visualization {

class ColormapFragmentShader
{
public:

  static const char* get()
  {
    static std::string foo(
                           "#version 410 core\n"
                           "\n"
                           "// Ouput data\n"
                           "out vec4 color;\n"
                           "\n"
                           "// Input data\n"
                           "in vec3 fragmentColor;\n"
                           "\n"
                           "void main()\n"
                           "{\n"
                           "// Output color\n"
                           "   color = vec4(fragmentColor,1.0f);\n"
                           "}\n"
                           );

    return foo.c_str();
  }
};

} // ns
} // ns

#endif
