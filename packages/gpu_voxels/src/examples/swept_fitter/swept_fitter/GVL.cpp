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
 * \author  Klaus Fischnaller
 * \date    2015-09-22
 *
 */
//----------------------------------------------------------------------
#include "GVL.h"
#include <sstream>


bool GVL::areColliding(std::string svol1, std::string svol2)
{
    // Create unique name for this collision
    std::stringstream ss;
    if (svol1 < svol2)
        ss << svol1 << "|" << svol2;
    else
        ss << svol2 << "|" << svol1;

    std::map<std::string, bool>::iterator col;

    std::string colName = ss.str();

    // check cache for already checked collision
    col = m_colmap.find(colName);

    if (col != m_colmap.end())
    {
        return col->second;
    }

    // Check for collision.
    // The same BitVector bit has to be set in order to detect a collision
    size_t numCols = m_gvl->getMap(svol1)->as<voxellist::BitVectorVoxelList>()->collideWithBitcheck(m_gvl->getMap(svol2)->as<voxellist::BitVectorVoxelList>());

    // cache result
    m_colmap[colName] = numCols != 0;

    return numCols != 0;
}

GVL::GVL()
{
    m_gvl = GpuVoxels::getInstance();
    m_gvl->initialize(300, 250, 100, 0.02);
}

GVL::~GVL()
{
    m_gvl.reset();
}


