// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the IC Workspace.
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
* \author  Felix Mauch <mauch@fzi.de>
* \date    2015-02-27
*
*/
//----------------------------------------------------------------------

#ifndef PERFORMANCEMONITORMACROS_H
#define PERFORMANCEMONITORMACROS_H

#ifdef IC_PERFORMANCE_MONITOR
  #define PERF_MON_INITIALIZE(num_names, num_events) \
    ::icl_core::perf_mon::PerformanceMonitor::initialize(num_names, num_events);
  #define PERF_MON_START(timer_name) \
    ::icl_core::perf_mon::PerformanceMonitor::start(timer_name);
  #define PERF_MON_ENABLE(prefix) \
    ::icl_core::perf_mon::PerformanceMonitor::enablePrefix(prefix);
  #define PERF_MON_DISABLE(prefix) \
    ::icl_core::perf_mon::PerformanceMonitor::disablePrefix(prefix);
  #define PERF_MON_ENABLE_ALL(enabled) \
    ::icl_core::perf_mon::PerformanceMonitor::enableAll(enabled);

  #define PERF_MON_PRINT_INFO(timer_name, description) \
    ::icl_core::perf_mon::PerformanceMonitor::measurement(timer_name, description, "", ::icl_core::logging::eLL_INFO);
  #define PERF_MON_PRINT_INFO_P(timer_name, description, prefix) \
    ::icl_core::perf_mon::PerformanceMonitor::measurement(timer_name, description, prefix, ::icl_core::logging::eLL_INFO);
  #define PERF_MON_PRINT_AND_RESET_INFO(timer_name, description) \
    ::icl_core::perf_mon::PerformanceMonitor::startStop(timer_name, description, "", ::icl_core::logging::eLL_INFO);
  #define PERF_MON_PRINT_AND_RESET_INFO_P(timer_name, description, prefix) \
    ::icl_core::perf_mon::PerformanceMonitor::startStop(timer_name, description, prefix, ::icl_core::logging::eLL_INFO);

  #define PERF_MON_ADD_STATIC_DATA(description, data) \
    ::icl_core::perf_mon::PerformanceMonitor::addStaticData(description, data, "");
  #define PERF_MON_ADD_STATIC_DATA_P(description, data, prefix) \
    ::icl_core::perf_mon::PerformanceMonitor::addStaticData(description, data, prefix);
  #define PERF_MON_ADD_DATA(description, data) \
    ::icl_core::perf_mon::PerformanceMonitor::addData(description, data, "");
  #define PERF_MON_ADD_DATA_P(description, data, prefix) \
    ::icl_core::perf_mon::PerformanceMonitor::addData(description, data, prefix);
  #define PERF_MON_ADD_DATA_NONTIME(description, data) \
    ::icl_core::perf_mon::PerformanceMonitor::addNonTimeData(description, data, "");
  #define PERF_MON_ADD_DATA_NONTIME_P(description, data, prefix) \
    ::icl_core::perf_mon::PerformanceMonitor::addNonTimeData(description, data, prefix);

  #define PERF_MON_SUMMARY_INFO(prefix, description) \
    ::icl_core::perf_mon::PerformanceMonitor::printSummary(prefix, description, ::icl_core::logging::eLL_INFO);
  #define PERF_MON_SUMMARY_PREFIX_INFO(prefix) \
    ::icl_core::perf_mon::PerformanceMonitor::printSummaryFromPrefix(prefix, ::icl_core::logging::eLL_INFO);
  #define PERF_MON_SUMMARY_ALL_INFO \
    ::icl_core::perf_mon::PerformanceMonitor::printSummaryAll(::icl_core::logging::eLL_INFO);


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
