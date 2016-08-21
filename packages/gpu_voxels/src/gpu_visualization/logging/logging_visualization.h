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
#ifndef LOGGING_VISUALIZATION_H_
#define LOGGING_VISUALIZATION_H_

#include <icl_core_logging/Logging.h>

namespace gpu_voxels {
namespace visualization {

DECLARE_LOG_STREAM(Visualization)
DECLARE_LOG_STREAM(Shader)
DECLARE_LOG_STREAM(SharedMemManager)
DECLARE_LOG_STREAM(Context)

using icl_core::logging::endl;
}
}

#endif /* LOGGING_CLASSIFICATION_H_ */
