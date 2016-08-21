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
#include "Trajectory.h"

Trajectory::Trajectory(GVL *gvl, std::string name, gpu_voxels::robot::JointValueMap min, gpu_voxels::robot::JointValueMap max)
    : m_gvl(gvl), m_name(name), m_min(min), m_max(max)
{
}

bool Trajectory::collidesWith(Trajectory *traj)
{
    return m_gvl->areColliding(traj->getName(), m_name);
}
