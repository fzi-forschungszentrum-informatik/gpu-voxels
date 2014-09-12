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
 * \date    2014-06-27
 *
 */
//----------------------------------------------------------------------
#include <gpu_voxels/logging/logging_octree.h>

namespace gpu_voxels {

REGISTER_LOG_STREAM(OctreeLog);
REGISTER_LOG_STREAM(OctreeDebugLog);
REGISTER_LOG_STREAM(OctreeDebugEXLog);
REGISTER_LOG_STREAM(OctreeFreespaceLog);
REGISTER_LOG_STREAM(OctreeInsertLog);
REGISTER_LOG_STREAM(OctreePropagateLog);
REGISTER_LOG_STREAM(OctreeCountBeforeExtractLog);
REGISTER_LOG_STREAM(OctreeExtractCubeLog);
REGISTER_LOG_STREAM(OctreeRebuildLog);
REGISTER_LOG_STREAM(OctreeFreeBoundingBoxLog);
REGISTER_LOG_STREAM(OctreeDepthCallbackLog);

}
