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
#ifndef FITTER_H
#define FITTER_H

#include <string>
#include <vector>
#include <list>
#include "Robot.h"
#include "GVL.h"

#include <iostream>

namespace SweptFitter {

struct Solution
{
    std::vector<std::vector<int> > solution;
    std::vector<std::list<int> > statesTodo;
    bool allSolutions;
};

class Fitter
{
public:
    Fitter(GVL *gvl);
    ~Fitter();
    Robot *createRobot(std::string name, std::string urdf);
    void fit(bool allSolutions = false);
    size_t getLastResultNum();

private:
    void initSolution(Solution &sol, bool allSolutions);
    void fitInternal(Solution &sol, size_t currentRobot, size_t index);
    bool collides(Solution &sol, int currentRobot, int index);
    void printSolution(Solution &sol);

    GVL *m_gvl;
    std::vector<Robot *> m_robots;
    size_t m_resultNum;
};


}

#endif // FITTER_H
