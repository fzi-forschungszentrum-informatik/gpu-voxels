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
#include "gvl_ompl_planner_helper.h"

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#include <gpu_voxels/GpuVoxels.h>
#include <gpu_voxels/helpers/MetaPointCloud.h>
#include <gpu_voxels/robot/urdf_robot/urdf_robot.h>
#include <gpu_voxels/logging/logging_gpu_voxels.h>
#include <thread>

#define IC_PERFORMANCE_MONITOR
#include <icl_core_performance_monitor/PerformanceMonitor.h>

namespace ob = ompl::base;
namespace og = ompl::geometric;
using namespace gpu_voxels;
namespace bfs = boost::filesystem;

GvlOmplPlannerHelper::GvlOmplPlannerHelper(const ob::SpaceInformationPtr &si)
    : ob::StateValidityChecker(si)
    , ob::MotionValidator(si)
{

    si_ = si;
    stateSpace_ = si_->getStateSpace().get();
    assert(stateSpace_ != nullptr);

    gvl = gpu_voxels::GpuVoxels::getInstance();
    gvl->initialize(150, 150, 100, 0.02);

    // We add maps with objects, to collide them
    gvl->addMap(MT_PROBAB_VOXELMAP,"myRobotMap");
    gvl->addMap(MT_PROBAB_VOXELMAP,"myEnvironmentMap");
    gvl->addMap(MT_BITVECTOR_VOXELLIST,"mySolutionMap");
    gvl->addMap(MT_PROBAB_VOXELMAP,"myQueryMap");

    gvl->addRobot("myUrdfRobot", "ur10_coarse/ur10_joint_limited_robot.urdf", true);

    gvl->visualizeMap("myEnvironmentMap");

    PERF_MON_ENABLE("pose_check");
    PERF_MON_ENABLE("motion_check");
    PERF_MON_ENABLE("motion_check_lv");
}



GvlOmplPlannerHelper::~GvlOmplPlannerHelper()
{
    gvl.reset(); // Not even required, as we use smart pointers.
}

void GvlOmplPlannerHelper::moveObstacle()
{
    gvl->clearMap("myEnvironmentMap");
    static float x(1.0);

    // use this to animate a single moving box obstacle
    //gvl->insertBoxIntoMap(Vector3f(2.0, x ,0.0), Vector3f(2.2, x + 0.2 ,1.2), "myEnvironmentMap", eBVM_OCCUPIED, 2);
    x += 0.1;

    gvl->insertBoxIntoMap(Vector3f(1.0,1.0,0.0), Vector3f(1.2,1.2,1.2), "myEnvironmentMap", eBVM_OCCUPIED, 2);
    gvl->insertBoxIntoMap(Vector3f(1.8,1.8,0.0), Vector3f(2.0,2.0,1.2), "myEnvironmentMap", eBVM_OCCUPIED, 2);
    gvl->insertBoxIntoMap(Vector3f(1.1,1.1,1.2), Vector3f(1.9,1.9,1.3), "myEnvironmentMap", eBVM_OCCUPIED, 2);
    gvl->insertBoxIntoMap(Vector3f(0.0,0.0,0.0), Vector3f(3.0,3.0,0.01), "myEnvironmentMap", eBVM_OCCUPIED, 2);
    gvl->visualizeMap("myEnvironmentMap");
}

void GvlOmplPlannerHelper::doVis()
{

    // tell the visualier that the map has changed:
    gvl->visualizeMap("myRobotMap");
    gvl->visualizeMap("myEnvironmentMap");
    gvl->visualizeMap("mySolutionMap");
    gvl->visualizeMap("myQueryMap");
}

void GvlOmplPlannerHelper::visualizeSolution(ob::PathPtr path)
{
    gvl->clearMap("mySolutionMap");

    PERF_MON_SUMMARY_PREFIX_INFO("pose_check");
    PERF_MON_SUMMARY_PREFIX_INFO("motion_check");
    PERF_MON_SUMMARY_PREFIX_INFO("motion_check_lv");

    std::cout << "Robot consists of " << gvl->getRobot("myUrdfRobot")->getTransformedClouds()->getAccumulatedPointcloudSize() << " points" << std::endl;

    og::PathGeometric* solution = path->as<og::PathGeometric>();
    solution->interpolate();


    for(size_t step = 0; step < solution->getStateCount(); ++step)
    {

        const double *values = solution->getState(step)->as<ob::RealVectorStateSpace::StateType>()->values;

        robot::JointValueMap state_joint_values;
        state_joint_values["shoulder_pan_joint"] = values[0];
        state_joint_values["shoulder_lift_joint"] = values[1];
        state_joint_values["elbow_joint"] = values[2];
        state_joint_values["wrist_1_joint"] = values[3];
        state_joint_values["wrist_2_joint"] = values[4];
        state_joint_values["wrist_3_joint"] = values[5];

        // update the robot joints:
        gvl->setRobotConfiguration("myUrdfRobot", state_joint_values);
        // insert the robot into the map:
        gvl->insertRobotIntoMap("myUrdfRobot", "mySolutionMap", BitVoxelMeaning(eBVM_SWEPT_VOLUME_START + (step % 249) ));
    }

    gvl->visualizeMap("mySolutionMap");

}

void GvlOmplPlannerHelper::insertStartAndGoal(const ompl::base::ScopedState<> &start, const ompl::base::ScopedState<> &goal) const
{

    gvl->clearMap("myQueryMap");

    robot::JointValueMap state_joint_values;
    state_joint_values["shoulder_pan_joint"] = start[0];
    state_joint_values["shoulder_lift_joint"] = start[1];
    state_joint_values["elbow_joint"] = start[2];
    state_joint_values["wrist_1_joint"] = start[3];
    state_joint_values["wrist_2_joint"] = start[4];
    state_joint_values["wrist_3_joint"] = start[5];

    // update the robot joints:
    gvl->setRobotConfiguration("myUrdfRobot", state_joint_values);
    gvl->insertRobotIntoMap("myUrdfRobot", "myQueryMap", BitVoxelMeaning(eBVM_SWEPT_VOLUME_START));

    state_joint_values["shoulder_pan_joint"] = goal[0];
    state_joint_values["shoulder_lift_joint"] = goal[1];
    state_joint_values["elbow_joint"] = goal[2];
    state_joint_values["wrist_1_joint"] = goal[3];
    state_joint_values["wrist_2_joint"] = goal[4];
    state_joint_values["wrist_3_joint"] = goal[5];

    // update the robot joints:
    gvl->setRobotConfiguration("myUrdfRobot", state_joint_values);
    gvl->insertRobotIntoMap("myUrdfRobot", "myQueryMap", BitVoxelMeaning(eBVM_SWEPT_VOLUME_START+1));

}

bool GvlOmplPlannerHelper::isValid(const ompl::base::State *state) const
{

    PERF_MON_START("inserting");

    std::lock_guard<std::mutex> lock(g_i_mutex);

    gvl->clearMap("myRobotMap");

    const double *values = state->as<ob::RealVectorStateSpace::StateType>()->values;

    robot::JointValueMap state_joint_values;
    state_joint_values["shoulder_pan_joint"] = values[0];
    state_joint_values["shoulder_lift_joint"] = values[1];
    state_joint_values["elbow_joint"] = values[2];
    state_joint_values["wrist_1_joint"] = values[3];
    state_joint_values["wrist_2_joint"] = values[4];
    state_joint_values["wrist_3_joint"] = values[5];

    // update the robot joints:
    gvl->setRobotConfiguration("myUrdfRobot", state_joint_values);
    // insert the robot into the map:
    gvl->insertRobotIntoMap("myUrdfRobot", "myRobotMap", eBVM_OCCUPIED);

    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("insert", "Pose Insertion", "pose_check");

    PERF_MON_START("coll_test");
    size_t num_colls_pc = gvl->getMap("myRobotMap")->as<voxelmap::ProbVoxelMap>()->collideWith(gvl->getMap("myEnvironmentMap")->as<voxelmap::ProbVoxelMap>(), 0.7f);
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("coll_test", "Pose Collsion", "pose_check");

    //std::cout << "Validity check on state ["  << values[0] << ", " << values[1] << ", " << values[2] << ", " << values[3] << ", " << values[4] << ", " << values[5] << "] resulting in " <<  num_colls_pc << " colls." << std::endl;

    return num_colls_pc == 0;
}

bool GvlOmplPlannerHelper::checkMotion(const ompl::base::State *s1, const ompl::base::State *s2,
                                       std::pair< ompl::base::State*, double > & lastValid) const
{

    //    std::cout << "LongestValidSegmentFraction = " << stateSpace_->getLongestValidSegmentFraction() << std::endl;
    //    std::cout << "getLongestValidSegmentLength = " << stateSpace_->getLongestValidSegmentLength() << std::endl;
    //    std::cout << "getMaximumExtent = " << stateSpace_->getMaximumExtent() << std::endl;


    std::lock_guard<std::mutex> lock(g_j_mutex);
    gvl->clearMap("myRobotMap");

    /* assume motion starts in a valid configuration so s1 is valid */

    bool result = true;
    int nd = stateSpace_->validSegmentCount(s1, s2);

    //std::cout << "Called interpolating motion_check_lv to evaluate " << nd << " segments" << std::endl;

    PERF_MON_ADD_DATA_NONTIME_P("Num poses in motion", float(nd), "motion_check_lv");
    if (nd > 1)
    {
        /* temporary storage for the checked state */
        ob::State *test = si_->allocState();

        for (int j = 1; j < nd; ++j)
        {
            stateSpace_->interpolate(s1, s2, (double)j / (double)nd, test);
            if (!si_->isValid(test))

            {
                lastValid.second = (double)(j - 1) / (double)nd;
                if (lastValid.first != nullptr)
                    stateSpace_->interpolate(s1, s2, lastValid.second, lastValid.first);
                result = false;
                break;

            }
        }

        si_->freeState(test);

    }

    if (result)
        if (!si_->isValid(s2))
        {
            lastValid.second = (double)(nd - 1) / (double)nd;
            if (lastValid.first != nullptr)
                stateSpace_->interpolate(s1, s2, lastValid.second, lastValid.first);
            result = false;
        }


    if (result)
        valid_++;
    else
        invalid_++;


    return result;
}

bool GvlOmplPlannerHelper::checkMotion(const ompl::base::State *s1, const ompl::base::State *s2) const
{
    std::lock_guard<std::mutex> lock(g_i_mutex);
    gvl->clearMap("myRobotMap");


    //        std::cout << "LongestValidSegmentFraction = " << stateSpace_->getLongestValidSegmentFraction() << std::endl;
    //        std::cout << "getLongestValidSegmentLength = " << stateSpace_->getLongestValidSegmentLength() << std::endl;
    //        std::cout << "getMaximumExtent = " << stateSpace_->getMaximumExtent() << std::endl;



    /* assume motion starts in a valid configuration so s1 is valid */

    bool result = true;
    int nd = stateSpace_->validSegmentCount(s1, s2);

    // not required with ProbabVoxels:
    //    if(nd > 249)
    //    {
    //        std::cout << "Too many intermediate states for BitVoxels" << std::endl;
    //        exit(1);
    //    }

    if (nd > 1)
    {
        PERF_MON_START("inserting");

        /* temporary storage for the checked state */
        ob::State *test = si_->allocState();

        for (int j = 1; j < nd; ++j)
        {
            stateSpace_->interpolate(s1, s2, (double)j / (double)nd, test);


            const double *values = test->as<ob::RealVectorStateSpace::StateType>()->values;

            robot::JointValueMap state_joint_values;
            state_joint_values["shoulder_pan_joint"] = values[0];
            state_joint_values["shoulder_lift_joint"] = values[1];
            state_joint_values["elbow_joint"] = values[2];
            state_joint_values["wrist_1_joint"] = values[3];
            state_joint_values["wrist_2_joint"] = values[4];
            state_joint_values["wrist_3_joint"] = values[5];

            // update the robot joints:
            gvl->setRobotConfiguration("myUrdfRobot", state_joint_values);
            // insert the robot into the map:
            gvl->insertRobotIntoMap("myUrdfRobot", "myRobotMap", eBVM_OCCUPIED);

        }
        PERF_MON_ADD_DATA_NONTIME_P("Num poses in motion", float(nd), "motion_check");

        si_->freeState(test);

        PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("insert", "Motion Insertion", "motion_check");

        //gvl->visualizeMap("myRobotMap");
        PERF_MON_START("coll_test");
        size_t num_colls_pc = gvl->getMap("myRobotMap")->as<voxelmap::ProbVoxelMap>()->collideWith(gvl->getMap("myEnvironmentMap")->as<voxelmap::ProbVoxelMap>(), 0.7f);
        //std::cout << "CheckMotion1 for " << nd << " segments. Resulting in " << num_colls_pc << " colls." << std::endl;
        PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("coll_test", "Pose Collsion", "motion_check");

        result = (num_colls_pc == 0);

    }


    if (result)
        valid_++;
    else
        invalid_++;


    return result;


}
