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
 * \date    2014-03-18
 *
 * \brief  Contains some defines for the visualizer.
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_VISUALIZATION_VISUALIZERDEFINES_H_INCLUDED
#define GPU_VOXELS_VISUALIZATION_VISUALIZERDEFINES_H_INCLUDED

namespace gpu_voxels {
namespace visualization {

//the first three values contain will contain the translation vector and the last value is the cube size
static const size_t SIZE_OF_TRANSLATION_VECTOR = 4*sizeof(float);

// if a buffer gets resized the new size will be : new_size + BUFFER_SIZE_FACTOR * new_size
static const float BUFFER_SIZE_FACTOR = 0.1f;

} //end of namespace visualization
} //end of namespace gpu_voxels

#endif
