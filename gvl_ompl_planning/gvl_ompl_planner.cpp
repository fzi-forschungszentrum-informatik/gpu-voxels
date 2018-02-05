// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
//
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE.txt in the top
// directory of the source code.
//
// © Copyright 2018 FZI Forschungszentrum Informatik, Karlsruhe, Germany
//
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Andreas Hermann
 * \date    2018-01-07
 *
 */
//----------------------------------------------------------------------/*
#include <iostream>
using namespace std;
#include <gpu_voxels/logging/logging_gpu_voxels.h>

#define IC_PERFORMANCE_MONITOR
#include <icl_core_performance_monitor/PerformanceMonitor.h>

#include <ompl/geometric/planners/sbl/SBL.h>
#include <ompl/geometric/planners/kpiece/LBKPIECE1.h>

#include <ompl/geometric/PathSimplifier.h>

#include "gvl_ompl_planner_helper.h"
#include <stdlib.h>

#include <memory>

namespace ob = ompl::base;
namespace og = ompl::geometric;




int main(int argc, char **argv)
{
    // Always initialize our logging framework as a first step, with the CLI arguments:
    icl_core::logging::initialize(argc, argv);

    PERF_MON_INITIALIZE(100, 1000);
    PERF_MON_ENABLE("planning");

    // construct the state space we are planning in
    auto space(std::make_shared<ob::RealVectorStateSpace>(6));
    //We then set the bounds for the R3 component of this state space:
    ob::RealVectorBounds bounds(6);
    bounds.setLow(-3.14159265);
    bounds.setHigh(3.14159265);

    bounds.setHigh(1, 0.0);

    space->setBounds(bounds);
    //Create an instance of ompl::base::SpaceInformation for the state space
    auto si(std::make_shared<ob::SpaceInformation>(space));
    //Set the state validity checker
    std::shared_ptr<GvlOmplPlannerHelper> my_class_ptr(std::make_shared<GvlOmplPlannerHelper>(si));

    og::PathSimplifier simp(si);

    si->setStateValidityChecker(my_class_ptr->getptr());
    si->setMotionValidator(my_class_ptr->getptr());
    si->setup();


    //Create a random start state:
    ob::ScopedState<> start(space);
    start[0] = -1.3;
    start[1] = -0.2;
    start[2] = 0.0;
    start[3] = 0.0;
    start[4] = 0.0;
    start[5] = 0.0;
    //And a random goal state:
    ob::ScopedState<> goal(space);
    goal[0] = 1.3;
    goal[1] = -0.5;
    goal[2] = 0.0;
    goal[3] = 0.0;
    goal[4] = 0.0;
    goal[5] = 0.0;

    my_class_ptr->insertStartAndGoal(start, goal);
    my_class_ptr->doVis();


    //Create an instance of ompl::base::ProblemDefinition
    auto pdef(std::make_shared<ob::ProblemDefinition>(si));
    //Set the start and goal states for the problem definition.
    pdef->setStartAndGoalStates(start, goal);
    //Create an instance of a planner
    auto planner(std::make_shared<og::LBKPIECE1>(si));
    //Tell the planner which problem we are interested in solving
    planner->setProblemDefinition(pdef);
    //Make sure all the settings for the space and planner are in order. This will also lead to the runtime computation of the state validity checking resolution.
    planner->setup();

    int succs = 0;

    //use this to stop planning, until Visualizer is connected
    std::cout << "Waiting for Viz. Press Key if ready!" << std::endl;
    std::cin.ignore();

    for(size_t n = 0; n < 5; ++n)
    {
        my_class_ptr->moveObstacle();

        //We can now try to solve the problem. This call returns a value from ompl::base::PlannerStatus which describes whether a solution has been found within the specified amount of time (in seconds). If this value can be cast to true, a solution was found.
        PERF_MON_START("planner");
        planner->clear(); // this clears all roadmaps
        ob::PlannerStatus solved = planner->ob::Planner::solve(20.0);
        PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("planner", "Planning time", "planning");

        //If a solution has been found, we simplify and display it.
        if (solved)
        {
            ++succs;
            // get the goal representation from the problem definition (not the same as the goal state)
            // and inquire about the found path
            ob::PathPtr path = pdef->getSolutionPath();
            std::cout << "Found solution:" << std::endl;
            // print the path to screen
            path->print(std::cout);

            PERF_MON_START("simplify");
            simp.simplifyMax(*(path->as<og::PathGeometric>()));
            PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("simplify", "Simplification time", "planning");

            std::cout << "Simplified solution:" << std::endl;
            // print the path to screen
            path->print(std::cout);

            my_class_ptr->visualizeSolution(path);
            usleep(500000);

        }else{
            std::cout << "No solution could be found" << std::endl;
        }

        PERF_MON_SUMMARY_PREFIX_INFO("planning");

    }
    PERF_MON_ADD_STATIC_DATA_P("Number of Planning Successes", succs, "planning");

    PERF_MON_SUMMARY_PREFIX_INFO("planning");

    // keep the visualization running:
    while(true)
    {
        my_class_ptr->doVis();
        usleep(30000);
    }

    return 1;
}
