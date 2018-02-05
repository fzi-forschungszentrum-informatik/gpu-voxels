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

#include "Fitter.h"
#include <iostream>
#include <setjmp.h>
static jmp_buf jmp;
using namespace SweptFitter;

Fitter::Fitter(GVL *gvl) : m_gvl(gvl), m_resultNum(0)
{
}

Fitter::~Fitter()
{
    std::vector<Robot*>::iterator i;
    for (i = m_robots.begin(); i != m_robots.end(); ++i)
    {
        delete *i;
    }
    m_robots.clear();
}

Robot *Fitter::createRobot(std::string name, std::string urdf)
{
    Robot *r = new Robot(m_gvl, name, urdf);
    m_robots.push_back(r);
    return r;
}

void Fitter::initSolution(Solution &sol, bool allSolutions)
{
    sol.allSolutions = allSolutions;
    sol.solution.clear();
    sol.statesTodo.clear();
    sol.solution.resize(m_robots.size());
    sol.statesTodo.resize(m_robots.size());

    for (size_t i = 0; i < m_robots.size(); i++)
    {
        sol.solution[i].clear();
        for (size_t j = 0; j < m_robots[i]->getNumTrajectories(); j++)
            sol.statesTodo[i].push_back(j);
    }
}

void Fitter::fit(bool allSolutions)
{
    if (allSolutions || !setjmp(jmp))
    {
        SweptFitter::Solution solution;
        initSolution(solution, allSolutions);
        fitInternal(solution, 0, 0);
    }
}

size_t Fitter::getLastResultNum()
{
    return m_resultNum;
}

void Fitter::fitInternal(Solution &sol, size_t currentRobot, size_t index)
{
    if (index >= m_robots[currentRobot]->getNumTrajectories())
    {
        if (currentRobot == m_robots.size() - 1)
        {
            printSolution(sol);
            m_resultNum++;
            if (!sol.allSolutions)
                longjmp(jmp,1);
        }
        else
        {
            fitInternal(sol, currentRobot+1, 0);
        }
        return;
    }

    for (size_t state = index; state < m_robots[currentRobot]->getNumTrajectories(); state++)
    {
        int traj = *(sol.statesTodo[currentRobot].begin());
        sol.statesTodo[currentRobot].pop_front();
        sol.solution[currentRobot].push_back(traj);

        if (!collides(sol, currentRobot, index))
        {
            fitInternal(sol, currentRobot, index+1);
        }
        sol.statesTodo[currentRobot].push_back(traj);
        sol.solution[currentRobot].pop_back();
    }
}

bool Fitter::collides(Solution &sol, int currentRobot, int index)
{
    int t = sol.solution[currentRobot][index];
    Trajectory *traj = m_robots[currentRobot]->getTrajectory(t);
    for (int r = currentRobot - 1; r >= 0; r--)
    {
        t = sol.solution[r][index];
        Trajectory *t2 = m_robots[r]->getTrajectory(t);
        if (traj->collidesWith(t2))
            return true;
    }
    return false;
}

void Fitter::printSolution(Solution &sol)
{
    //return;
    std::cout << "-------------------" << std::endl;
    for (size_t r = 0; r < sol.solution.size(); r++)
    {
        std::cout << m_robots[r]->getName() << ":  ";
        for (size_t i = 0; i < sol.solution[r].size(); i++)
        {
            Trajectory *t = m_robots[r]->getTrajectory(sol.solution[r][i]);
            std::cout << t->getName() << " ";
        }
        std::cout << std::endl;
    }
}
