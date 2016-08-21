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
#ifndef TRAJECTORY_H
#define TRAJECTORY_H

#include <string>
#include "GVL.h"
#include <gpu_voxels/robot/urdf_robot/urdf_robot.h>

class Trajectory
{
public:
    Trajectory(GVL *gvl, std::string name, gpu_voxels::robot::JointValueMap min, gpu_voxels::robot::JointValueMap max);

    bool collidesWith(Trajectory *traj);
    std::string getName() { return m_name; }

    gpu_voxels::robot::JointValueMap &getMin() { return m_min; }
    gpu_voxels::robot::JointValueMap &getMax() { return m_max; }

private:
    GVL *m_gvl;
    std::string m_name;
    gpu_voxels::robot::JointValueMap m_min;
    gpu_voxels::robot::JointValueMap m_max;
};

#endif // TRAJECTORY_H
