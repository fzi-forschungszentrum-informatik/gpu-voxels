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
 * \date    2014-06-23
 *
 */
//----------------------------------------------------------------------
#include <gpu_visualization/logging/logging_visualization.h>

namespace gpu_voxels {
namespace visualization {

REGISTER_LOG_STREAM (Visualization);
REGISTER_LOG_STREAM (Shader);
REGISTER_LOG_STREAM (SharedMemManager);
REGISTER_LOG_STREAM (Context);

}
}
