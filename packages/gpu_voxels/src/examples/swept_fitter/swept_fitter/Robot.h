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
#ifndef ROBOT_H
#define ROBOT_H

#include <string>
#include <vector>

#include "GVL.h"
#include "Trajectory.h"

class Robot
{
public:
    Robot(GVL *gvl, std::string name, std::string urdf);
    ~Robot();

    std::string getName() { return m_name; }
    void loadTrajectories(std::string path, size_t numTraj, bool useModelPath=true);
    Trajectory *getTrajectory(size_t index) { return m_trajectories[index]; }
    size_t getNumTrajectories() { return m_trajectories.size(); }
    void renderSweptVolumes();

private:
    GVL *m_gvl;
    std::string m_name;
    std::string m_urdf;
    std::vector<Trajectory*> m_trajectories;
};

#endif // ROBOT_H
