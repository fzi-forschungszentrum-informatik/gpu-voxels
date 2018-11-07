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
* \author  Felix Mauch <mauch@fzi.de>
* \date    2015-02-27
*
*/
//----------------------------------------------------------------------

#ifndef PERFORMANCEMONITORMACROS_H
#define PERFORMANCEMONITORMACROS_H

/**
 * @brief These macros help using the performance monitor. Most macros are defined for INFO, DEBUG
 * and TRACE logging level. DEBUG and TRACE are ignored by the compiler if IC_DEBUG is not defined.
 *
 * Timers can be started in prefixes which can be switched on and off separately. If no prefix is
 * given, the timers will be started in the defaule prefix "". All macros that use a prefix end on
 * _P
 *
 * All custom prefixes have to be enabled with the PERF_MON_ENABLE macro, or all of them can be enabled
 * with PERF_MON_ENABLE_ALL.
 * Prefixes can be disabled with the PERF_MON_DISABLE again.
 *
 * The documentation in this file is very brief. If you want further information about the performance
 * monitor, please see the header for the performance monitor class.
 */

#ifdef IC_PERFORMANCE_MONITOR
  //! Initializes the performance monitor with an approximate number of names and events. These numbers
  //! don't have to be exact, they only define the size of the underlying data structures.
  #define PERF_MON_INITIALIZE(num_names, num_events) \
    ::icl_core::perf_mon::PerformanceMonitor::initialize(num_names, num_events);

  //! start a timer with a given identifier. The timer can be accessed with this identifier later.
  #define PERF_MON_START(timer_name) \
    ::icl_core::perf_mon::PerformanceMonitor::start(timer_name);

  //! enables all timers under the given prefix.
  #define PERF_MON_ENABLE(prefix) \
    ::icl_core::perf_mon::PerformanceMonitor::enablePrefix(prefix);
  //! disables all timers under the given prefix
  #define PERF_MON_DISABLE(prefix) \
    ::icl_core::perf_mon::PerformanceMonitor::disablePrefix(prefix);
  //! enables all prefixes at once
  #define PERF_MON_ENABLE_ALL(enabled) \
    ::icl_core::perf_mon::PerformanceMonitor::enableAll(enabled);

  //! Adds a measurement event to the given timer and prints it's time to the screen plus the given description
  #define PERF_MON_PRINT_INFO(timer_name, description) \
    ::icl_core::perf_mon::PerformanceMonitor::measurement(timer_name, description, "", ::icl_core::logging::eLL_INFO);
  //! Print measurement with given prefix
  #define PERF_MON_PRINT_INFO_P(timer_name, description, prefix) \
    ::icl_core::perf_mon::PerformanceMonitor::measurement(timer_name, description, prefix, ::icl_core::logging::eLL_INFO);

  //! Performs a time measurement, prints its time on the screen and resets the timer to 0
  #define PERF_MON_PRINT_AND_RESET_INFO(timer_name, description) \
    ::icl_core::perf_mon::PerformanceMonitor::startStop(timer_name, description, "", ::icl_core::logging::eLL_INFO);
  //! Time measurement with reset and prefix
  #define PERF_MON_PRINT_AND_RESET_INFO_P(timer_name, description, prefix) \
    ::icl_core::perf_mon::PerformanceMonitor::startStop(timer_name, description, prefix, ::icl_core::logging::eLL_INFO);

  //! Performs a time measurement and resets the timer to 0
  #define PERF_MON_SILENT_MEASURE_AND_RESET_INFO(timer_name, description) \
    ::icl_core::perf_mon::PerformanceMonitor::startStop(timer_name, description, "", ::icl_core::logging::eLL_INFO, true);
  //! Time measurement with reset and prefix
  #define PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P(timer_name, description, prefix) \
    ::icl_core::perf_mon::PerformanceMonitor::startStop(timer_name, description, prefix, ::icl_core::logging::eLL_INFO, true);

  //! add arbitrary single floating point data. This information will be listed in de summary as well
  #define PERF_MON_ADD_STATIC_DATA(description, data) \
    ::icl_core::perf_mon::PerformanceMonitor::addStaticData(description, data, "");
  //! add arbitrary single floating point data. This information will be listed in de summary as well with a given prefix
  #define PERF_MON_ADD_STATIC_DATA_P(description, data, prefix) \
    ::icl_core::perf_mon::PerformanceMonitor::addStaticData(description, data, prefix);
  //! Manually add a time measurement under the given description
  #define PERF_MON_ADD_DATA(description, data) \
    ::icl_core::perf_mon::PerformanceMonitor::addData(description, data, "");
  //! Manually add a time measurement under the given description and a prefix
  #define PERF_MON_ADD_DATA_P(description, data, prefix) \
    ::icl_core::perf_mon::PerformanceMonitor::addData(description, data, prefix);

  //! Add arbitrary floating point data to the statistis. Multiple measurements are allowed
  #define PERF_MON_ADD_DATA_NONTIME(description, data) \
    ::icl_core::perf_mon::PerformanceMonitor::addNonTimeData(description, data, "");
  //! Add arbitrary floating point data to the statistis. Multiple measurements are allowed (with prefix)
  #define PERF_MON_ADD_DATA_NONTIME_P(description, data, prefix) \
    ::icl_core::perf_mon::PerformanceMonitor::addNonTimeData(description, data, prefix);

  //! Print summary of all timers from a given prefix and the given description
  #define PERF_MON_SUMMARY_INFO(prefix, description) \
    ::icl_core::perf_mon::PerformanceMonitor::printSummary(prefix, description, ::icl_core::logging::eLL_INFO);
  //! Print summary of all events from a given prefix
  #define PERF_MON_SUMMARY_PREFIX_INFO(prefix) \
    ::icl_core::perf_mon::PerformanceMonitor::printSummaryFromPrefix(prefix, ::icl_core::logging::eLL_INFO);
  //! Print summary of all occured events
  #define PERF_MON_SUMMARY_ALL_INFO \
    ::icl_core::perf_mon::PerformanceMonitor::printSummaryAll(::icl_core::logging::eLL_INFO);


  // The following macros are the same as above, but for the DEBUG and TRACE case.
  #ifdef _IC_DEBUG_
    #define PERF_MON_PRINT_DEBUG(timer_name, description) \
      ::icl_core::perf_mon::PerformanceMonitor::measurement(timer_name, description, "", ::icl_core::logging::eLL_DEBUG);
    #define PERF_MON_PRINT_DEBUG_P(timer_name, description, prefix) \
      ::icl_core::perf_mon::PerformanceMonitor::measurement(timer_name, description, prefix, ::icl_core::logging::eLL_DEBUG);
    #define PERF_MON_PRINT_TRACE(timer_name, description) \
      ::icl_core::perf_mon::PerformanceMonitor::measurement(timer_name, description, "", ::icl_core::logging::eLL_TRACE);
    #define PERF_MON_PRINT_TRACE_P(timer_name, description, prefix) \
      ::icl_core::perf_mon::PerformanceMonitor::measurement(timer_name, description, prefix, ::icl_core::logging::eLL_TRACE);
    #define PERF_MON_PRINT_AND_RESET_DEBUG(timer_name, description) \
      ::icl_core::perf_mon::PerformanceMonitor::startStop(timer_name, description, "", ::icl_core::logging::eLL_DEBUG);
    #define PERF_MON_PRINT_AND_RESET_DEBUG_P(timer_name, description, prefix) \
      ::icl_core::perf_mon::PerformanceMonitor::startStop(timer_name, description, prefix, ::icl_core::logging::eLL_DEBUG);
    #define PERF_MON_PRINT_AND_RESET_TRACE(timer_name, description) \
      ::icl_core::perf_mon::PerformanceMonitor::startStop(timer_name, description, "", ::icl_core::logging::eLL_TRACE);
    #define PERF_MON_PRINT_AND_RESET_TRACE_P(timer_name, description, prefix) \
      ::icl_core::perf_mon::PerformanceMonitor::startStop(timer_name, description, prefix, ::icl_core::logging::eLL_TRACE);

    #define PERF_MON_SILENT_MEASURE_AND_RESET_DEBUG(timer_name, description) \
      ::icl_core::perf_mon::PerformanceMonitor::startStop(timer_name, description, "", ::icl_core::logging::eLL_DEBUG, true);
    #define PERF_MON_SILENT_MEASURE_AND_RESET_DEBUG_P(timer_name, description, prefix) \
      ::icl_core::perf_mon::PerformanceMonitor::startStop(timer_name, description, prefix, ::icl_core::logging::eLL_DEBUG, true);
    #define PERF_MON_SILENT_MEASURE_AND_RESET_TRACE(timer_name, description) \
      ::icl_core::perf_mon::PerformanceMonitor::startStop(timer_name, description, "", ::icl_core::logging::eLL_TRACE, true);
    #define PERF_MON_SILENT_MEASURE_AND_RESET_TRACE_P(timer_name, description, prefix) \
      ::icl_core::perf_mon::PerformanceMonitor::startStop(timer_name, description, prefix, ::icl_core::logging::eLL_TRACE, true);

    #define PERF_MON_SUMMARY_DEBUG(prefix, description) \
      ::icl_core::perf_mon::PerformanceMonitor::printSummary(prefix, description, ::icl_core::logging::eLL_DEBUG);
    #define PERF_MON_SUMMARY_TRACE(prefix, description) \
      ::icl_core::perf_mon::PerformanceMonitor::printSummary(prefix, description, ::icl_core::logging::eLL_TRACE);
    #define PERF_MON_SUMMARY_PREFIX_DEBUG(prefix) \
      ::icl_core::perf_mon::PerformanceMonitor::printSummaryFromPrefix(prefix, ::icl_core::logging::eLL_DEBUG);
    #define PERF_MON_SUMMARY_PREFIX_TRACE(prefix) \
      ::icl_core::perf_mon::PerformanceMonitor::printSummaryFromPrefix(prefix, ::icl_core::logging::eLL_TRACE);
    #define PERF_MON_SUMMARY_ALL_DEBUG \
      ::icl_core::perf_mon::PerformanceMonitor::printSummaryAll(::icl_core::logging::eLL_DEBUG);
    #define PERF_MON_SUMMARY_ALL_TRACE \
      ::icl_core::perf_mon::PerformanceMonitor::printSummaryAll(::icl_core::logging::eLL_TRACE);
  #else
    #define PERF_MON_PRINT_DEBUG(timer_name, description) (void)0
    #define PERF_MON_PRINT_DEBUG_P(timer_name, description, prefix) (void)0
    #define PERF_MON_PRINT_TRACE(timer_name, description) (void)0
    #define PERF_MON_PRINT_TRACE_P(timer_name, description, prefix) (void)0
    #define PERF_MON_PRINT_AND_RESET_DEBUG(timer_name, description) (void)0
    #define PERF_MON_PRINT_AND_RESET_DEBUG_P(timer_name, description, prefix) (void)0
    #define PERF_MON_PRINT_AND_RESET_TRACE(timer_name, description) (void)0
    #define PERF_MON_PRINT_AND_RESET_TRACE_P(timer_name, description, prefix) (void)0

    #define PERF_MON_SILENT_MEASURE_AND_RESET_DEBUG(timer_name, description) (void)0
    #define PERF_MON_SILENT_MEASURE_AND_RESET_DEBUG_P(timer_name, description, prefix) (void)0
    #define PERF_MON_SILENT_MEASURE_AND_RESET_TRACE(timer_name, description) (void)0
    #define PERF_MON_SILENT_MEASURE_AND_RESET_TRACE_P(timer_name, description, prefix) (void)0

    #define PERF_MON_SUMMARY_DEBUG(prefix, description) (void)0
    #define PERF_MON_SUMMARY_TRACE(prefix, description) (void)0
    #define PERF_MON_SUMMARY_PREFIX_DEBUG(prefix) (void)0
    #define PERF_MON_SUMMARY_PREFIX_TRACE(prefix) (void)0
    #define PERF_MON_SUMMARY_ALL_DEBUG(prefix, description) (void)0
    #define PERF_MON_SUMMARY_ALL_TRACE(prefix, description) (void)0
  #endif


#else
  #define PERF_MON_INITIALIZE(num_names, num_events) (void)0
  #define PERF_MON_START(timer_name) (void)0
  #define PERF_MON_ENABLE(prefix) (void)0
  #define PERF_MON_DISABLE(prefix) (void)0
  #define PERF_MON_ENABLE_ALL(enabled) (void)0
  #define PERF_MON_SUMMARY(prefix, description) (void)0
  #define PERF_MON_PRINT_INFO(timer_name, description) (void)0
  #define PERF_MON_PRINT_INFO_P(timer_name, description, prefix) (void)0
  #define PERF_MON_PRINT_DEBUG(timer_name, description) (void)0
  #define PERF_MON_PRINT_DEBUG_P(timer_name, description, prefix) (void)0
  #define PERF_MON_PRINT_TRACE(timer_name, description) (void)0
  #define PERF_MON_PRINT_TRACE_P(timer_name, description, prefix) (void)0
  #define PERF_MON_PRINT_AND_RESET_INFO(timer_name, description) (void)0
  #define PERF_MON_PRINT_AND_RESET_INFO_P(timer_name, description, prefix) (void)0
  #define PERF_MON_PRINT_AND_RESET_DEBUG(timer_name, description) (void)0
  #define PERF_MON_PRINT_AND_RESET_DEBUG_P(timer_name, description, prefix) (void)0
  #define PERF_MON_PRINT_AND_RESET_TRACE(timer_name, description) (void)0
  #define PERF_MON_PRINT_AND_RESET_TRACE_P(timer_name, description, prefix) (void)0

  #define PERF_MON_SILENT_MEASURE_AND_RESET_INFO(timer_name, description) (void)0
  #define PERF_MON_SILENT_MEASURE_AND_RESET_DEBUG(timer_name, description) (void)0
  #define PERF_MON_SILENT_MEASURE_AND_RESET_TRACE(timer_name, description) (void)0

  #define PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P(timer_name, description, prefix) (void)0
  #define PERF_MON_SILENT_MEASURE_AND_RESET_DEBUG_P(timer_name, description, prefix) (void)0
  #define PERF_MON_SILENT_MEASURE_AND_RESET_TRACE_P(timer_name, description, prefix) (void)0

  #define PERF_MON_SUMMARY_INFO(prefix, description) (void)0
  #define PERF_MON_SUMMARY_DEBUG(prefix, description) (void)0
  #define PERF_MON_SUMMARY_TRACE(prefix, description) (void)0
  #define PERF_MON_SUMMARY_PREFIX_INFO(prefix) (void)0
  #define PERF_MON_SUMMARY_PREFIX_DEBUG(prefix) (void)0
  #define PERF_MON_SUMMARY_PREFIX_TRACE(prefix) (void)0
  #define PERF_MON_SUMMARY_ALL_INFO (void)0
  #define PERF_MON_SUMMARY_ALL_DEBUG (void)0
  #define PERF_MON_SUMMARY_ALL_TRACE (void)0
  #define PERF_MON_ADD_STATIC_DATA(description, data) (void)0
  #define PERF_MON_ADD_STATIC_DATA_P(description, data, prefix) (void)0
  #define PERF_MON_ADD_DATA(description, data) (void)0
  #define PERF_MON_ADD_DATA_P(description, data, prefix) (void)0
  #define PERF_MON_ADD_DATA_NONTIME(description, data) (void)0
  #define PERF_MON_ADD_DATA_NONTIME_P(description, data, prefix) (void)0
#endif

#endif // PERFORMANCEMONITORMACROS_H
