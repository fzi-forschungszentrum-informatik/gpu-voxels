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
#ifndef LOGGING_OCTREE_H_
#define LOGGING_OCTREE_H_

#include <icl_core_logging/Logging.h>
namespace gpu_voxels {

DECLARE_LOG_STREAM(OctreeLog)
DECLARE_LOG_STREAM(OctreeDebugLog)
DECLARE_LOG_STREAM(OctreeDebugEXLog)
DECLARE_LOG_STREAM(OctreeFreespaceLog)
DECLARE_LOG_STREAM(OctreeInsertLog)
DECLARE_LOG_STREAM(OctreePropagateLog)
DECLARE_LOG_STREAM(OctreeCountBeforeExtractLog)
DECLARE_LOG_STREAM(OctreeExtractCubeLog)
DECLARE_LOG_STREAM(OctreeRebuildLog)
DECLARE_LOG_STREAM(OctreeFreeBoundingBoxLog)
DECLARE_LOG_STREAM(OctreeDepthCallbackLog)
using icl_core::logging::endl;
}
#endif /* LOGGING_OCTREE_H_ */
