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
 * \author  Felix Mauch <mauch@fzi.de>
 * \date    2015-02-12
 *
 */
//----------------------------------------------------------------------/*

#ifndef ICL_CORE_TIMER_H
#define ICL_CORE_TIMER_H

#include <string>
#include <vector>
#include <map>
#include <ostream>

#include "icl_core_performance_monitor/logging_performance_monitor.h"
#include "icl_core_performance_monitor/PerformanceMonitorMacros.h"
#include "icl_core_performance_monitor/ImportExport.h"


#include <icl_core/TimeStamp.h>

namespace icl_core {
namespace perf_mon{


/**
 * @brief The PerformanceMonitor class provides an easy to use tool for performance
 * measurement.
 *
 * You can use the static methods directly, but using the macros defined in
 * \a PerformanceMonitorMacros.h is recommended, as they provide enabling and disabling by
 * defines such as _IC_DEBUG_ and IC_PERFORMANCE_MONITOR.
 *
 * The typical minimum workflow would look like follows:
 *  - In your main do:
 *   - enable PM macros with #define IC_PERFORMANCE_MONITOR (before including this header!!!)
 *   - initialize the PM with number of names and events
 *   - at the end of your program print the summary
 *  - Somewhere in your program do:
 *   - start a timer
 *   - do some work
 *   - create measurement event (this will by default output the runtime
 *
 * By default all outputs go to the prefix "". You can enable other prefixes as you wish
 * and then use the print and summary functions with these prefixes. This way you can
 * easily turn different groups on and off.
 *
 * Static information such as parallel configurations can be passed to the performance
 * monitor as well, which will be printed in the summary, as well.
 */
class ICL_CORE_PERFORMANCE_MONITOR_EXPORT PerformanceMonitor
{
public:
  /**
   * @brief getInstance As The Performance Monitor is implemented as a singleton pattern,
   * this provides an accessor to the PM instance. If there's no instance, yet, one will
   * be created.
   * @return pointer to the PM instance
   */
  static PerformanceMonitor* getInstance();

  /**
   * @brief initialize Initialize the performance monitor. Call this before using anything
   * from PerformanceMonitor
   * @param num_names Assumed maximum number of event names (descriptions). This serves as
   * preallocation of the according map.
   * @param num_events Assumed maximum number of events (timer measurements). This serves as
   * preallocation of the according map.
   */
  static void initialize(const uint32_t num_names, const uint32_t num_events);

  /**
   * @brief start Start a timer with the given identifier
   * @param timer_name The timer's identifier
   */
  static void start(std::string timer_name);

  /**
   * @brief measurement Make a measurement from a given timer. The timer will keep running
   * and the measurement will be added to PM's event log under the given description.
   * @param timer_name The timer's identifier
   * @param description The event description. Keep it short for better readability.
   * @param prefix Optional prefix to put the event into a group. Defaults to ""
   * @param level Optional logging level. Defaults to icl_core::logging::eLL_INFO
   * @return The measurement value in ms
   */
  static double measurement(std::string timer_name, std::string description, std::string prefix = "",
                          logging::LogLevel level = icl_core::logging::eLL_INFO);

  /**
   * @brief measurement Make a measurement from a given timer. Resets the timer
   * and the measurement will be added to PM's event log under the given description.
   * If the given timer isn't started yet, it just will be started.
   * @param timer_name The timer's identifier
   * @param description The event description. Keep it short for better readability.
   * @param prefix Optional prefix to put the event into a group. Defaults to ""
   * @param level Optional logging level. Defaults to icl_core::logging::eLL_INFO
   * @param silent Optional overwrite to suppress the output into the logstream
   * @return The measurement value in ms
   */
  static double startStop(std::string timer_name, std::string description, std::string prefix = "",
                          logging::LogLevel level = icl_core::logging::eLL_INFO, const bool silent = false);

  /**
   * @brief addData Manually insert a time measurement with given identifier
   * @param name Timer description
   * @param data Time
   * @param prefix Prefix that the data belongs to.
   */
  static void addData(std::string name, double data, std::string prefix);

  /**
   * @brief addStaticData Adds static information. You can basically put any information
   * that can be cast into a double in here.
   * @param name Description of the data
   * @param data The data itself
   * @param prefix Prefix that the data belongs to.
   */
  static void addStaticData(std::string name, double data, std::string prefix);

  /**
   * @brief addNonTimeData Adds some additional arbitrary information. In contrast to addStaticData
   * multiple values (measurements) can be passed to the same identifier. However, right now
   * they are output as if they were time measurements.
   * @param name Short data description
   * @param data The data itself
   * @param prefix Prefix that the data belongs to.
   */
  static void addNonTimeData(std::string name, double data, std::string prefix);

  /**
   * @brief getData Returns all data added under the given prefix and name combination.
   * @param name Short data description
   * @param prefix Prefix that the data belongs to.
   */
  static std::vector<double> getData(std::string name, std::string prefix);

  /**
   * @brief getNonTimeData Returns all nontime data added under the given prefix and name combination.
   * @param name Short data description
   * @param prefix Prefix that the data belongs to.
   */
  static std::vector<double> getNonTimeData(std::string name, std::string prefix);

  /**
   * @brief enablePrefix Enables a given prefix
   * @param prefix Prefix that will be enabled.
   */
  static void enablePrefix(std::string prefix);

  /**
   * @brief enableAll set whether all prefixes should be enabled or not. Overrides single
   * enable/disable assignments.
   *
   * Once enableAll(false) is called, single enable/disables statements will be used again.
   */
  static void enableAll(const bool& enabled);

  /**
   * @brief disablePrefix Disables a given prefix
   * @param prefix Prefix that will be disabled.
   */
  static void disablePrefix(std::string prefix);

  /**
   * @brief printSummary Print a summary for a given collection of events.
   * @param prefix The prefix in which the events lie.
   * @param name The events' description
   * @param level Optional logging level. Defaults to icl_core::logging::eLL_INFO
   */
  static std::string printSummary(std::string prefix, std::string name,
                           icl_core::logging::LogLevel level = icl_core::logging::eLL_INFO);

  /**
   * @brief printSummaryAll Print summary of everything: All prefixes, all descriptions
   * and all static data. Output will be sorted by prefixes and static data will be printed
   * first.
   * @param level Optional logging level. Defaults to icl_core::logging::eLL_INFO
   */
  static std::string printSummaryAll(icl_core::logging::LogLevel level = icl_core::logging::eLL_INFO);


  /**
   * @brief printSummaryFromPrefix Prints summary for the given prefix only.
   * @param prefix Prefix for which the summary will be printed
   * @param level Optional logging level. Defaults to icl_core::logging::eLL_INFO
   */
  static std::string printSummaryFromPrefix(std::string prefix, icl_core::logging::LogLevel level = icl_core::logging::eLL_INFO);

  //! if set to false, the performance monitor will be non-operational
  bool m_enabled;
  //! if set to false, the performance monitor won't print measurement events
  bool m_print_stop;

protected:
  //! constructor
  PerformanceMonitor();
  //! destructor
  ~PerformanceMonitor();

  //! Create output string for summary
  void createStatisticSummary(std::stringstream& ss, std::string prefix, std::string name);
  void createStatisticSummaryNonTime(std::stringstream& ss, std::string prefix, std::string name);


  //! prints the given message to the specified log level
  void print(std::string message, icl_core::logging::LogLevel level = icl_core::logging::eLL_DEBUG);

  //! add event to the event map
  void addEvent(std::string prefix, std::string name, double data);
  void addNonTimeEvent(std::string prefix, std::string name, double data);

  //! check if prefix is enabled
  bool isEnabled(std::string prefix);

  //! calculate the average of all events with description name
  double getAverage(std::string name);
  double getAverageNonTime(std::string name);

  //! calculate the median of all events with description name
  void getMedian(std::string name, double& median, double& min, double& max);
  void getMedianNonTime(std::string name, double& median, double& min, double& max);

  std::map<std::string, std::vector<double> > m_data;
  std::map<std::string, std::vector<double> > m_data_nontime;
  std::map<std::string, TimeStamp> m_timer;
  std::vector<std::vector<double> > m_buffer;
  std::map<std::string, bool> m_enabled_prefix;
  std::map<std::string, double > m_static_data;

  bool m_all_enabled;


private:
  static PerformanceMonitor* m_instance;
};


} // namespace perf_mon
} // namespace icl_core

#endif /* ICL_CORE_TIMER_H */
