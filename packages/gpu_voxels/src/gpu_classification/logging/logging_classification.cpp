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
#include <gpu_classification/logging/logging_classification.h>

namespace gpu_voxels {
namespace classification {

REGISTER_LOG_STREAM(Classification);
REGISTER_LOG_STREAM(GroundSegmentation);
REGISTER_LOG_STREAM(ObjectSegmentation);
REGISTER_LOG_STREAM(SegmentationHelp);
REGISTER_LOG_STREAM(Filter);
REGISTER_LOG_STREAM(DecisionTree);
REGISTER_LOG_STREAM(Kinect);
REGISTER_LOG_STREAM(TestExample);

}
}
