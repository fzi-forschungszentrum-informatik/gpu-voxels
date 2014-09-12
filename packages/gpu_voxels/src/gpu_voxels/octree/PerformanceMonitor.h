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
 * \date    2014-13-21
 *
 */
//----------------------------------------------------------------------/*

#ifndef GPU_VOXELS_OCTREE_PERFORMANCE_MONITOR_H_INCLUDED
#define GPU_VOXELS_OCTREE_PERFORMANCE_MONITOR_H_INCLUDED

#include <string>
#include <vector>
#include <map>
#include <ostream>

namespace gpu_voxels {
namespace NTree {

class PerformanceMonitor
{
public:
  static PerformanceMonitor* getInstance();

  void init(int num_names, int num_events);

  static void start(std::string timer_name);

  static double stop(std::string timer_name, std::string prefix, std::string name);

  static void addData(std::string prefix, std::string name, double data);

  static void addStaticData(std::string prefix, std::string name, double data);

  void printSummary(std::ostream &stream);

  void printHeader(std::ostream &stream);

  void enable(std::string prefix);

  void disable(std::string prefix);

  bool m_enabled;
  bool m_all_enabled;
  bool m_logging;
  bool m_print_stop;
  bool m_print_addData;

  static const std::string separator;

protected:
  PerformanceMonitor();
  ~PerformanceMonitor();

  void addEvent(std::string prefix, std::string name, double data);

  bool isEnabled(std::string prefix);

  double getAverage(std::string name);

  void getMedian(std::string name, double& median, double& min, double& max);

  std::map<std::string, std::vector<double> > m_data;
  std::map<std::string, timespec> m_timer;
  std::vector<std::vector<double> > m_buffer;
  std::map<std::string, bool> m_enabled_prefix;
  std::map<std::string, double > m_static_data;


private:
  static PerformanceMonitor* m_instance;
};

}
}

#endif /* PERFORMANCEMONITOR_H_ */
