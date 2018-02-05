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

#include <iostream>
using namespace std;
#include <gpu_voxels/logging/logging_gpu_voxels.h>
#include "swept_fitter/GVL.h"
#include "swept_fitter/Fitter.h"
#include "swept_fitter/Robot.h"
#include <boost/timer.hpp>
#include <stdlib.h>

const struct {
    std::string name;
    std::string urdf;
    std::string trajectories;
} roboTemplates[] = {
    {"HoLLiE_1","hollie/hollie.urdf","hollie.traj"},
    {"HoLLiE_2","hollie/hollie.urdf","hollie2.traj"},
    {"HoLLiE_3","hollie/hollie.urdf","hollie3.traj"}
};

int main(int argc, char **argv)
{
    icl_core::logging::initialize(argc, argv);

    GVL gvl;

    bool allResults = true;
    size_t numTraj = 3;
    size_t numRobots = sizeof(roboTemplates) / sizeof(roboTemplates[0]);

    SweptFitter::Fitter fitter(&gvl);

    boost::timer timer;

    // Create robots
    for (size_t i = 0; i < numRobots; i++)
    {
        Robot *r = fitter.createRobot(roboTemplates[i].name, roboTemplates[i].urdf);
        r->loadTrajectories(roboTemplates[i].trajectories, numTraj);
        r->renderSweptVolumes();
    }

    double init_time = timer.elapsed();

    // Uncomment this to visualize the robot poses
    // while creating a new one

    //while(1) { usleep(10000);}

    timer.restart();
    fitter.fit(allResults);
    double fitting_time = timer.elapsed();

    std::cout << std::endl << "== STATISTICS ==" << std::endl
              << "results: " << (allResults? "all":"first")
                    << " (" << fitter.getLastResultNum() << " found)" << std::endl
              << "robots: " << numRobots << std::endl
              << "trajectories: " << numTraj << std::endl
              << "init time: " << init_time << " s" << std::endl
              << "fitting time: " << fitting_time << " s" << std::endl
              << "overall time: " << (init_time+fitting_time) << " s" << std::endl;

    return 0;
}

