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
#ifndef LOGGING_CLASSIFICATION_H_
#define LOGGING_CLASSIFICATION_H_

#include <icl_core_logging/Logging.h>

namespace gpu_voxels {
namespace classification {

DECLARE_LOG_STREAM(Classification)
DECLARE_LOG_STREAM(GroundSegmentation)
DECLARE_LOG_STREAM(ObjectSegmentation)
DECLARE_LOG_STREAM(SegmentationHelp)
DECLARE_LOG_STREAM(Filter)
DECLARE_LOG_STREAM(DecisionTree)
DECLARE_LOG_STREAM(Kinect);
DECLARE_LOG_STREAM(TestExample);

using icl_core::logging::endl;
}
}

#endif /* LOGGING_CLASSIFICATION_H_ */
