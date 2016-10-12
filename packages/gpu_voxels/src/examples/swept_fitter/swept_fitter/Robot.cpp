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
#include "Robot.h"
#include <iostream>
#include <fstream>
#include <gpu_voxels/robot/urdf_robot/urdf_robot.h>
#include <gpu_voxels/helpers/PointcloudFileHandler.h>
#include <sstream>

Robot::Robot(GVL *gvl, std::string name, std::string urdf)
    : m_gvl(gvl), m_name(name), m_urdf(urdf)
{
}

Robot::~Robot()
{
    std::vector<Trajectory*>::iterator i;
    for (i = m_trajectories.begin(); i != m_trajectories.end(); ++i)
    {
        delete *i;
    }
    m_trajectories.clear();
}

void Robot::loadTrajectories(std::string path, size_t numTraj, bool useModelPath)
{
    path = (getGpuVoxelsPath(useModelPath) / boost::filesystem::path("trajectories") / boost::filesystem::path(path)).string();

    std::ifstream trajs(path.c_str(), std::ifstream::in);

    if (!trajs.good())
    {
        std::cerr << "Could not open file " << path << std::endl;
        return;
    }

    size_t num;

    std::string input;
    trajs >> input;

    if (input != "Trajectory_Num:")
    {
        std::cerr << "Illegal file format." << std::endl;
        return;
    }

    trajs >> num;

    Trajectory *t;
    for (size_t i = 0; i < num && i < numTraj; i++)
    {
        trajs >> input;
        if (input != "Joint_Num:")
        {
            std::cerr << "Illegal file format." << std::endl;
            return;
        }

        int jointNum;
        trajs >> jointNum;

        trajs >> input;

        if (input != "Name:")
        {
            std::cerr << "Illegal file format." << std::endl;
            return;
        }

        std::string traj_name;
        trajs >> traj_name;

        gpu_voxels::robot::JointValueMap _min;
        gpu_voxels::robot::JointValueMap _max;

        for (int j = 0; j < jointNum; j++)
        {
            std::string name;
            double min;
            double max;
            trajs >> name;
            trajs >> min;
            trajs >> max;

            _min[name] = min;
            _max[name] = max;
        }

        t = new Trajectory(m_gvl, traj_name, _min, _max);
        m_trajectories.push_back(t);
    }
}

void Robot::renderSweptVolumes()
{
    GpuVoxelsSharedPtr gvl = m_gvl->getGVL();
    std::cout << "adding robot " << m_name << std::endl;
    gvl->addRobot(m_name, m_urdf, true);

    std::vector<Trajectory*>::iterator i;

    // Render all trajectories into VoxelLists
    for (i = m_trajectories.begin(); i != m_trajectories.end(); ++i)
    {
        Trajectory *t = *i;

        gvl->addMap(MT_BITVECTOR_VOXELLIST, t->getName());


        // Use 100 intermediate poses
        size_t intermPoses = 100;
        for (size_t i = 0; i <= intermPoses; i += 1)
        {
            // In this example the motion is determined by
            // linear interpolation between start and end poses.
            gpu_voxels::robot::JointValueMap jointValues = interpolateLinear(t->getMin(),
                                                                             t->getMax(),
                                                                             (1.0/intermPoses) * i);

            gvl->setRobotConfiguration(m_name, jointValues);
            BitVoxelMeaning vt = (BitVoxelMeaning)(eBVM_SWEPT_VOLUME_START + i);
            gvl->insertRobotIntoMap(m_name, t->getName(), vt);

        }

    }

}
