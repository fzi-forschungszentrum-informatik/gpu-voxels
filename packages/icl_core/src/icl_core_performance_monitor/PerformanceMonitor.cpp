// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE in the top
// directory of the source code.
//
// © Copyright 2018 FZI Forschungszentrum Informatik, Karlsruhe, Germany
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Florian Drews
 * \date    2013-11-16
 *
 */
//----------------------------------------------------------------------/*

#include "icl_core_performance_monitor/PerformanceMonitor.h"

#include <stdlib.h>
#include <algorithm>
#include <sstream>

#include <icl_core/TimeSpan.h>

using namespace std;

namespace icl_core {
namespace perf_mon{

string makeName(string prefix, string name)
{
  return prefix + "::" + name;
}

PerformanceMonitor* PerformanceMonitor::m_instance = NULL;

PerformanceMonitor::PerformanceMonitor()
{
  m_enabled = true;
  m_print_stop = true;
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

void PerformanceMonitor::initialize(const uint32_t num_names, const uint32_t num_events)
{
  PerformanceMonitor* monitor = getInstance();
  monitor->m_data.clear();
  monitor->m_data_nontime.clear();
  monitor->m_buffer.resize(num_names);
  for (uint32_t i = 0; i < num_names; ++i)
    monitor->m_buffer[i].reserve(num_events);

  monitor->enablePrefix("");
}


bool PerformanceMonitor::isEnabled(string prefix)
{
  return (m_enabled && m_enabled_prefix.find(prefix) != m_enabled_prefix.end()) || m_all_enabled;
}

void PerformanceMonitor::start(string timer_name)
{
  if (getInstance()->m_enabled)
  {
    TimeStamp t = TimeStamp::now();
    getInstance()->m_timer[timer_name] = t;
  }
}

double PerformanceMonitor::measurement(string timer_name, string description, string prefix,
                               icl_core::logging::LogLevel level)
{
  PerformanceMonitor* monitor = getInstance();
  if (monitor->isEnabled(prefix))
  {
    TimeStamp end = TimeStamp::now();
    TimeSpan d(end - monitor->m_timer[timer_name]);
    double double_ms = d.toNSec() / 1000000.0;
    monitor->addEvent(prefix, description, double_ms);

    if (getInstance()->m_print_stop)
    {
      std::stringstream ss;
      ss << makeName(prefix, description) << ": " << double_ms << " ms";
      monitor->print(ss.str(), level);
    }
    return double_ms;
  }
  return 0;
}

double PerformanceMonitor::startStop(string timer_name, string description, string prefix,
                                     logging::LogLevel level, const bool silent)
{
  /*
   * If timer_name exists:
   *   stop timer
   *   make new start time equal to stop time
   * else
   *   start timer
   */

  if (getInstance()->isEnabled(prefix))
  {
    PerformanceMonitor* monitor = getInstance();
    TimeStamp start = monitor->m_timer[timer_name];
    if (start != TimeStamp())
    {
      TimeStamp end = TimeStamp::now();
      TimeSpan d(end - start);
      double double_ms = d.toNSec() / 1000000.0;
      monitor->addEvent(prefix, description, double_ms);
      monitor->m_timer[timer_name] = end;
      if (!silent && getInstance()->m_print_stop)
      {
        std::stringstream ss;
        ss << makeName(prefix, description) << ": " << double_ms << " ms";
        monitor->print(ss.str(), level);
      }
      return double_ms;
    }
    else
    {
      PerformanceMonitor::start(timer_name);
    }
  }
  return 0;
}

void PerformanceMonitor::addStaticData(string name, double data, string prefix)
{
  if (getInstance()->isEnabled(prefix))
  {
    string tmp = makeName(prefix, name);
    getInstance()->m_static_data[tmp] = data;
  }
}

void PerformanceMonitor::addData(string name, double data, string prefix)
{
  if (getInstance()->isEnabled(prefix))
  {
    getInstance()->addEvent(prefix, name, data);
  }
}

void PerformanceMonitor::addNonTimeData(string name, double data, string prefix)
{
  if (getInstance()->isEnabled(prefix))
  {
    getInstance()->addNonTimeEvent(prefix, name, data);
  }
}

void PerformanceMonitor::addEvent(string prefix, string name, double data)
{
  if (getInstance()->isEnabled(prefix))
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

void PerformanceMonitor::addNonTimeEvent(string prefix, string name, double data)
{
  if (getInstance()->isEnabled(prefix))
  {
    string tmp = makeName(prefix, name);
    if (m_data_nontime.find(tmp) == m_data_nontime.end())
    {
      m_data_nontime[tmp] = vector<double>();
      if (m_buffer.size() > 0)
      {
        m_data_nontime[tmp].swap(m_buffer.back());
        m_buffer.pop_back();
      }
    }
    m_data_nontime[tmp].push_back(data);
  }
}


void PerformanceMonitor::print(string message, logging::LogLevel level)
{
  switch (level)
  {
    case ::icl_core::logging::eLL_DEBUG:
    {
      LOGGING_DEBUG(Performance, message << endl);
      break;
    }
    case ::icl_core::logging::eLL_INFO:
    {
      LOGGING_INFO(Performance, message << endl);
      break;
    }
    case ::icl_core::logging::eLL_TRACE:
    {
      LOGGING_TRACE(Performance, message << endl);
      break;
    }
    default:
    {
      LOGGING_INFO(Performance, message << endl);
      break;
    }
  }
}

void PerformanceMonitor::createStatisticSummary(stringstream& ss, string prefix, string name)
{
  string tmp = makeName(prefix, name);
  double median, min, max;
  getMedian(tmp, median, min, max);

  ss << "Summary for " << tmp << "\n" <<
        "Called " << m_data[tmp].size() << " times\n" <<
        name << "_avg: " << getAverage(tmp) << " ms\n" <<
        name << "_median: " << median << " ms\n" <<
        name << "_min: " << min << " ms\n" <<
        name << "_max: " << max << " ms\n"<<
        "\n";
}

void PerformanceMonitor::createStatisticSummaryNonTime(stringstream& ss, string prefix, string name)
{
  string tmp = makeName(prefix, name);
  double median, min, max;
  getMedianNonTime(tmp, median, min, max);

  ss << "Summary for " << tmp << "\n" <<
        "num entries: " << m_data_nontime[tmp].size() << "\n" <<
        name << "_avg: " << getAverageNonTime(tmp) << "\n" <<
        name << "_median: " << median << "\n" <<
        name << "_min: " << min << "\n" <<
        name << "_max: " << max << "\n"<<
        "\n";
}

string PerformanceMonitor::printSummary(string prefix, string name,
                                      icl_core::logging::LogLevel level)
{
  PerformanceMonitor* monitor = getInstance();

  std::stringstream ss;
  monitor->createStatisticSummary(ss, prefix, name);
  monitor->print(ss.str(), level);
  return ss.str();
}

void PerformanceMonitor::enablePrefix(string prefix)
{
  PerformanceMonitor* monitor = getInstance();

  if (monitor->m_enabled_prefix.find(prefix) == monitor->m_enabled_prefix.end())
  {
    monitor->m_enabled_prefix[prefix] = true;
  }
}

void PerformanceMonitor::enableAll(const bool& enabled)
{
  getInstance()->m_all_enabled = enabled;
}

void PerformanceMonitor::disablePrefix(string prefix)
{
  PerformanceMonitor* monitor = getInstance();

  if (monitor->m_enabled_prefix.find(prefix) != monitor->m_enabled_prefix.end())
  {
    monitor->m_enabled_prefix.erase(prefix);
  }
}

string PerformanceMonitor::printSummaryAll(icl_core::logging::LogLevel level)
{
  PerformanceMonitor* monitor = getInstance();

  std::stringstream ss;
  for (map<string, bool>::iterator it=monitor->m_enabled_prefix.begin();
       it != monitor->m_enabled_prefix.end(); ++it)
  {
    ss << printSummaryFromPrefix(it->first, level);
  }
  return ss.str();
}

string PerformanceMonitor::printSummaryFromPrefix(string prefix, icl_core::logging::LogLevel level)
{
  PerformanceMonitor* monitor = getInstance();
  bool first = true;
  std::stringstream ss;
  ss << "\n########## Begin of Summary for prefix " << prefix << " ##########\n";
  for (map<string, double>::iterator it = monitor->m_static_data.begin(); it != monitor->m_static_data.end(); it++)
  {
    size_t prefix_end = it->first.find("::");
    std::string prefix_tmp = it->first.substr(0, prefix_end);

    if (prefix == prefix_tmp)
    {
      if (first)
      {
        ss << "#### Static data: ####\n";
        first = false;
      }
      ss <<  it->first.substr(prefix_end+2) << ": " << it->second << "\n";
    }
  }

  first = true;
  for (map<string, vector<double> >::iterator it = monitor->m_data.begin(); it != monitor->m_data.end(); it++)
  {
    size_t prefix_end = it->first.find("::");
    std::string prefix_tmp = it->first.substr(0, prefix_end);

    if (prefix == prefix_tmp)
    {
      if (first)
      {
        ss << "#### Time data: ####\n";
        first = false;
      }
      string name = it->first.substr(prefix_end+2);
      monitor->createStatisticSummary(ss, prefix, name);
    }
  }

  first = true;
  for (map<string, vector<double> >::iterator it = monitor->m_data_nontime.begin(); it != monitor->m_data_nontime.end(); it++)
  {
    size_t prefix_end = it->first.find("::");
    std::string prefix_tmp = it->first.substr(0, prefix_end);

    if (prefix == prefix_tmp)
    {
      if (first)
      {
        ss << "#### Non-time data: ####\n";
        first = false;
      }
      string name = it->first.substr(prefix_end+2);
      monitor->createStatisticSummaryNonTime(ss, prefix, name);
    }
  }
  monitor->print(ss.str(), level);
  return ss.str();
}

double PerformanceMonitor::getAverage(string name)
{
  double avg=0;
  vector<double>* tmp = &m_data[name];
  int n = (int) m_data[name].size();
  for (int i = 0; i < n; ++i)
    avg = avg + tmp->at(i);
  avg = avg / n;
  return avg;
}

double PerformanceMonitor::getAverageNonTime(string name)
{
  double avg=0;
  vector<double>* tmp = &m_data_nontime[name];
  int n = (int) m_data_nontime[name].size();
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

void PerformanceMonitor::getMedianNonTime(string name, double& median, double& min, double& max)
{
  vector<double> tmp = m_data_nontime[name];
  sort(tmp.begin(), tmp.end());
  if (tmp.size() > 0)
  median = tmp[tmp.size() / 2];
  min = tmp[0];
  max = tmp[tmp.size() - 1];
}

vector<double> PerformanceMonitor::getData(string name, string prefix)
{
  PerformanceMonitor* monitor = getInstance();
  return monitor->m_data[ makeName(prefix, name) ];
}

vector<double> PerformanceMonitor::getNonTimeData(string name, string prefix)
{
  PerformanceMonitor* monitor = getInstance();
  return monitor->m_data_nontime[ makeName(prefix, name) ];
}

} // namespace timer
} // namespace icl_core

