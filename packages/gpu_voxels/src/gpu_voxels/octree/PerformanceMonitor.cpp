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
 * \author  Florian Drews
 * \date    2013-11-16
 *
 */
//----------------------------------------------------------------------/*

#include "PerformanceMonitor.h"
#include <stdlib.h>
#include <algorithm>
#include <stdio.h>
#include <gpu_voxels/octree/DataTypes.h>

using namespace std;

namespace gpu_voxels {
namespace NTree {

const string PerformanceMonitor::separator = "\t";
PerformanceMonitor* PerformanceMonitor::m_instance = NULL;

PerformanceMonitor::PerformanceMonitor()
{
  m_enabled = true;
  m_logging = true;
  m_print_stop = true;
  m_print_addData = true;
  m_all_enabled = false;
}

PerformanceMonitor::~PerformanceMonitor()
{

}

PerformanceMonitor* PerformanceMonitor::getInstance()
{
  if (m_instance == NULL)
  {
    m_instance = new PerformanceMonitor();
  }
  return m_instance;
}

string makeName(string prefix, string name)
{
  return prefix + "::" + name;
}

bool PerformanceMonitor::isEnabled(string prefix)
{
  return (m_enabled && m_enabled_prefix.find(prefix) != m_enabled_prefix.end()) || m_all_enabled;
}

void PerformanceMonitor::init(int num_names, int num_events)
{
  m_data.clear();
  m_buffer.resize(num_names);
  for (int i = 0; i < num_names; ++i)
    m_buffer[i].reserve(num_events);
}

void PerformanceMonitor::start(string timer_name)
{
  if (getInstance()->m_enabled)
  {
    timespec t = getCPUTime();
    getInstance()->m_timer[timer_name] = t;
  }
}

double PerformanceMonitor::stop(string timer_name, string prefix, string name)
{
  if (getInstance()->isEnabled(prefix))
  {
    timespec t = getCPUTime();
    double d = timeDiff(getInstance()->m_timer[timer_name], t);
    getInstance()->addEvent(prefix, name, d);
    if (getInstance()->m_print_stop)
      printf("%s %f ms\n", makeName(prefix, name).c_str(), d);
    return d;
  }
  return 0;
}

void PerformanceMonitor::addStaticData(string prefix, string name, double data)
{
  if (getInstance()->isEnabled(prefix))
  {
    if (getInstance()->m_logging)
    {
      string tmp = makeName(prefix, name);
      getInstance()->m_static_data[tmp] = data;
    }
    if (getInstance()->m_print_addData)
      printf("%s %f\n", makeName(prefix, name).c_str(), data);
  }
}

void PerformanceMonitor::addData(string prefix, string name, double data)
{
  if (getInstance()->isEnabled(prefix))
  {
    getInstance()->addEvent(prefix, name, data);
    if (getInstance()->m_print_addData)
      printf("%s %f\n", makeName(prefix, name).c_str(), data);
  }
}

void PerformanceMonitor::addEvent(string prefix, string name, double data)
{
  if (getInstance()->isEnabled(prefix))
  {
    if (m_logging)
    {
      string tmp = makeName(prefix, name);
      if (m_data.find(tmp) == m_data.end())
      {
        m_data[tmp] = vector<double>();
        if (m_buffer.size() > 0)
        {
          m_data[tmp].swap(m_buffer.back());
          m_buffer.pop_back();
        }
      }
      m_data[tmp].push_back(data);
    }
  }
}

void PerformanceMonitor::enable(string prefix)
{
  if (m_enabled_prefix.find(prefix) == m_enabled_prefix.end())
  {
    m_enabled_prefix[prefix] = true;
  }
}

void PerformanceMonitor::disable(string prefix)
{
  if (m_enabled_prefix.find(prefix) != m_enabled_prefix.end())
  {
    m_enabled_prefix.erase(prefix);
  }
}

void PerformanceMonitor::printHeader(ostream &stream)
{
  // print header
  for (map<string, double>::iterator it = m_static_data.begin(); it != m_static_data.end(); it++)
    stream << it->first << separator;

  for (map<string, vector<double> >::iterator it = m_data.begin(); it != m_data.end(); it++)
    stream << it->first << "_avg" << separator << it->first << "_median" << separator
        << it->first << "_min" << separator << it->first << "_max" << separator;

  stream << endl;
}

void PerformanceMonitor::printSummary(ostream &stream)
{
  for (map<string, double>::iterator it = m_static_data.begin(); it != m_static_data.end(); it++)
  {
    stream << it->second << separator;
  }
  for (map<string, vector<double> >::iterator it = m_data.begin(); it != m_data.end(); it++)
  {
    double median, min, max;
    getMedian(it->first, median, min, max);

    stream << getAverage(it->first) << separator << median << separator << min << separator << max
        << separator;
  }
  stream << endl;
}

double PerformanceMonitor::getAverage(string name)
{
  double avg;
  vector<double>* tmp = &m_data[name];
  int n = (int) m_data[name].size();
  for (int i = 0; i < n; ++i)
    avg = avg + tmp->at(i);
  avg = avg / n;
  return avg;
}

void PerformanceMonitor::getMedian(string name, double& median, double& min, double& max)
{
  vector<double> tmp = m_data[name];
  sort(tmp.begin(), tmp.end());
  median = tmp[tmp.size() / 2];
  min = tmp[0];
  max = tmp[tmp.size() - 1];
}

}
}

