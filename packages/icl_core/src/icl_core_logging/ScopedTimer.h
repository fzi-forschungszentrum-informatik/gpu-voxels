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
 * \author  Jan Oberlaender <oberlaender@fzi.de>
 * \date    2013-12-13
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_LOGGING_SCOPED_TIMER_H_INCLUDED
#define ICL_CORE_LOGGING_SCOPED_TIMER_H_INCLUDED

#include <boost/preprocessor/cat.hpp>

#include "icl_core/TimeStamp.h"
#include "icl_core/TimeSpan.h"
#include "icl_core_logging/LogLevel.h"
#include "icl_core_logging/LogStream.h"
#include "icl_core_logging/ThreadStream.h"
#include "icl_core_logging/LoggingMacros_SLOGGING.h"

namespace icl_core {
namespace logging {

/*! A helper class for measuring the time spent in a scope.  While you
 *  can use this class directly, you should prefer the
 *  LOGGING_SCOPED_TIMER* macros instead.
 *  \tparam TStreamName Name of the LogStream class to write to.
 */
template <typename TStreamName>
class ScopedTimer
{
public:
  /*! Create a scoped timer.  Upon destruction, it outputs the time in
   *  nanoseconds that has passed since construction to the \a
   *  TStreamName LogStream.  The \a description is output alongside
   *  to help identify which timer is presented.
   *  \param description Description of the timer.  If, for example,
   *         description is "Foo" and destruction occurs after 42
   *         nanoseconds, the output is "Foo: 42 ns".
   *  \param level The LogLevel.
   *  \param filename Name of the source file in which the timer was
   *         created.
   *  \param line Number of the source code line in which the timer
   *         was created.
   *  \param classname Name of the class in which the timer was
   *         created.
   *  \param objectname Name of the concrete object in which the timer
   *         was created.
   */
  ScopedTimer(const std::string& description = "ScopedTimer",
              const LogLevel level = eLL_DEBUG,
              const std::string& filename = __FILE__,
              const std::size_t line = __LINE__,
              const std::string& classname = "",
              const std::string& objectname = "")
    : m_description(description),
      m_level(level),
      m_filename(filename),
      m_line(line),
      m_classname(classname),
      m_objectname(objectname),
      m_start_time(TimeStamp::now()),
      m_active(true)
  { }

  //! Outputs the time passed since construction.
  void print() const
  {
    ::icl_core::logging::LogStream& stream = TStreamName::instance();
    SLOGGING_LOG_FLCO(stream, m_level, m_filename.c_str(), m_line, m_classname.c_str(), m_objectname.c_str(),
                      "" << m_description << ": "
                      << (TimeStamp::now() - m_start_time).toNSec() << " ns" << endl);
  }

  //! Outputs the time passed since construction.
  void print(const std::string& extra_description) const
  {
    ::icl_core::logging::LogStream& stream = TStreamName::instance();
    SLOGGING_LOG_FLCO(stream, m_level, m_filename.c_str(), m_line, m_classname.c_str(), m_objectname.c_str(),
                      "" << m_description << " (" << extra_description << "): "
                      << (TimeStamp::now() - m_start_time).toNSec() << " ns" << endl);
  }

  /*! Outputs the time passed since construction and stops the timer.
   *  The time is not output if stop() was previously called.
   */
  inline void stop()
  {
    if (m_active)
    {
      print();
    }
    m_active = false;
  }

  /*! Outputs the time passed since construction and destroys the
   *  timer.  The time is not output if stop() was previously called.
   */
  ~ScopedTimer()
  {
    if (m_active)
    {
      print();
    }
  }

private:
  const std::string m_description;
  const LogLevel m_level;
  const std::string m_filename;
  const std::size_t m_line;
  const std::string m_classname;
  const std::string m_objectname;
  const TimeStamp m_start_time;
  bool m_active;
};

}
}

#define LOGGING_SCOPED_TIMER_VFLCO(streamname, varname, description, level, filename, line, classname, objectname) \
  ::icl_core::logging::ScopedTimer<streamname> varname(description, level, filename, line, classname, objectname)
#define LOGGING_SCOPED_TIMER_VCO(streamname, varname, description, level, classname, objectname) \
  LOGGING_SCOPED_TIMER_VFLCO(streamname, varname, description, level, __FILE__, __LINE__, classname, objectname)
#define LOGGING_SCOPED_TIMER_VC(streamname, varname, description, level, classname) \
  LOGGING_SCOPED_TIMER_VFLCO(streamname, varname, description, level, __FILE__, __LINE__, classname, "")
#define LOGGING_SCOPED_TIMER_V(streamname, varname, description, level) \
  LOGGING_SCOPED_TIMER_VFLCO(streamname, varname, description, level, __FILE__, __LINE__, "", "")

#define LOGGING_SCOPED_TIMER_FLCO(streamname, description, level, filename, line, classname, objectname) \
  LOGGING_SCOPED_TIMER_VFLCO(streamname, BOOST_PP_CAT(scoped_timer_, line), description, level, filename, line, classname, objectname)
#define LOGGING_SCOPED_TIMER_CO(streamname, description, level, classname, objectname) \
  LOGGING_SCOPED_TIMER_FLCO(streamname, description, level, __FILE__, __LINE__, classname, objectname)
#define LOGGING_SCOPED_TIMER_C(streamname, description, level, classname) \
  LOGGING_SCOPED_TIMER_FLCO(streamname, description, level, __FILE__, __LINE__, classname, "")
#define LOGGING_SCOPED_TIMER(streamname, description, level) \
  LOGGING_SCOPED_TIMER_FLCO(streamname, description, level, __FILE__, __LINE__, "", "")

#define LOGGING_SCOPED_TIMER_ERROR(streamname, description)   LOGGING_SCOPED_TIMER(streamname, description, ::icl_core::logging::eLL_ERROR)
#define LOGGING_SCOPED_TIMER_WARNING(streamname, description) LOGGING_SCOPED_TIMER(streamname, description, ::icl_core::logging::eLL_WARNING)
#define LOGGING_SCOPED_TIMER_INFO(streamname, description)    LOGGING_SCOPED_TIMER(streamname, description, ::icl_core::logging::eLL_INFO)
#ifdef _IC_DEBUG_
# define LOGGING_SCOPED_TIMER_DEBUG(streamname, description)   LOGGING_SCOPED_TIMER(streamname, description, ::icl_core::logging::eLL_DEBUG)
# define LOGGING_SCOPED_TIMER_TRACE(streamname, description)   LOGGING_SCOPED_TIMER(streamname, description, ::icl_core::logging::eLL_TRACE)
#else
# define LOGGING_SCOPED_TIMER_DEBUG(streamname, description) (void)0
# define LOGGING_SCOPED_TIMER_TRACE(streamname, description) (void)0
#endif

#define LOGGING_SCOPED_TIMER_ERROR_C(streamname, description, classname)   LOGGING_SCOPED_TIMER_C(streamname, description, ::icl_core::logging::eLL_ERROR,   classname)
#define LOGGING_SCOPED_TIMER_WARNING_C(streamname, description, classname) LOGGING_SCOPED_TIMER_C(streamname, description, ::icl_core::logging::eLL_WARNING, classname)
#define LOGGING_SCOPED_TIMER_INFO_C(streamname, description, classname)    LOGGING_SCOPED_TIMER_C(streamname, description, ::icl_core::logging::eLL_INFO,    classname)
#ifdef _IC_DEBUG_
# define LOGGING_SCOPED_TIMER_DEBUG_C(streamname, description, classname)   LOGGING_SCOPED_TIMER_C(streamname, description, ::icl_core::logging::eLL_DEBUG,   classname)
# define LOGGING_SCOPED_TIMER_TRACE_C(streamname, description, classname)   LOGGING_SCOPED_TIMER_C(streamname, description, ::icl_core::logging::eLL_TRACE,   classname)
#else
# define LOGGING_SCOPED_TIMER_DEBUG_C(streamname, description, classname) (void)0
# define LOGGING_SCOPED_TIMER_TRACE_C(streamname, description, classname) (void)0
#endif

#define LOGGING_SCOPED_TIMER_ERROR_CO(streamname, description, classname, objectname)   LOGGING_SCOPED_TIMER_CO(streamname, description, ::icl_core::logging::eLL_ERROR,   classname, objectname)
#define LOGGING_SCOPED_TIMER_WARNING_CO(streamname, description, classname, objectname) LOGGING_SCOPED_TIMER_CO(streamname, description, ::icl_core::logging::eLL_WARNING, classname, objectname)
#define LOGGING_SCOPED_TIMER_INFO_CO(streamname, description, classname, objectname)    LOGGING_SCOPED_TIMER_CO(streamname, description, ::icl_core::logging::eLL_INFO,    classname, objectname)
#ifdef _IC_DEBUG_
# define LOGGING_SCOPED_TIMER_DEBUG_CO(streamname, description, classname, objectname)   LOGGING_SCOPED_TIMER_CO(streamname, description, ::icl_core::logging::eLL_DEBUG,   classname, objectname)
# define LOGGING_SCOPED_TIMER_TRACE_CO(streamname, description, classname, objectname)   LOGGING_SCOPED_TIMER_CO(streamname, description, ::icl_core::logging::eLL_TRACE,   classname, objectname)
#else
# define LOGGING_SCOPED_TIMER_DEBUG_CO(streamname, description, classname, objectname) (void)0
# define LOGGING_SCOPED_TIMER_TRACE_CO(streamname, description, classname, objectname) (void)0
#endif

#define LOGGING_SCOPED_TIMER_ERROR_V(streamname, varname, description)   LOGGING_SCOPED_TIMER_V(streamname, varname, description, ::icl_core::logging::eLL_ERROR)
#define LOGGING_SCOPED_TIMER_WARNING_V(streamname, varname, description) LOGGING_SCOPED_TIMER_V(streamname, varname, description, ::icl_core::logging::eLL_WARNING)
#define LOGGING_SCOPED_TIMER_INFO_V(streamname, varname, description)    LOGGING_SCOPED_TIMER_V(streamname, varname, description, ::icl_core::logging::eLL_INFO)
#ifdef _IC_DEBUG_
# define LOGGING_SCOPED_TIMER_DEBUG_V(streamname, varname, description)   LOGGING_SCOPED_TIMER_V(streamname, varname, description, ::icl_core::logging::eLL_DEBUG)
# define LOGGING_SCOPED_TIMER_TRACE_V(streamname, varname, description)   LOGGING_SCOPED_TIMER_V(streamname, varname, description, ::icl_core::logging::eLL_TRACE)
#else
# define LOGGING_SCOPED_TIMER_DEBUG_V(streamname, varname, description) (void)0
# define LOGGING_SCOPED_TIMER_TRACE_V(streamname, varname, description) (void)0
#endif

#define LOGGING_SCOPED_TIMER_ERROR_VC(streamname, varname, description, classname)   LOGGING_SCOPED_TIMER_VC(streamname, varname, description, ::icl_core::logging::eLL_ERROR,   classname)
#define LOGGING_SCOPED_TIMER_WARNING_VC(streamname, varname, description, classname) LOGGING_SCOPED_TIMER_VC(streamname, varname, description, ::icl_core::logging::eLL_WARNING, classname)
#define LOGGING_SCOPED_TIMER_INFO_VC(streamname, varname, description, classname)    LOGGING_SCOPED_TIMER_VC(streamname, varname, description, ::icl_core::logging::eLL_INFO,    classname)
#ifdef _IC_DEBUG_
# define LOGGING_SCOPED_TIMER_DEBUG_VC(streamname, varname, description, classname)   LOGGING_SCOPED_TIMER_VC(streamname, varname, description, ::icl_core::logging::eLL_DEBUG,   classname)
# define LOGGING_SCOPED_TIMER_TRACE_VC(streamname, varname, description, classname)   LOGGING_SCOPED_TIMER_VC(streamname, varname, description, ::icl_core::logging::eLL_TRACE,   classname)
#else
# define LOGGING_SCOPED_TIMER_DEBUG_VC(streamname, varname, description, classname) (void)0
# define LOGGING_SCOPED_TIMER_TRACE_VC(streamname, varname, description, classname) (void)0
#endif

#define LOGGING_SCOPED_TIMER_ERROR_VCO(streamname, varname, description, classname, objectname)   LOGGING_SCOPED_TIMER_VCO(streamname, varname, description, ::icl_core::logging::eLL_ERROR,   classname, objectname)
#define LOGGING_SCOPED_TIMER_WARNING_VCO(streamname, varname, description, classname, objectname) LOGGING_SCOPED_TIMER_VCO(streamname, varname, description, ::icl_core::logging::eLL_WARNING, classname, objectname)
#define LOGGING_SCOPED_TIMER_INFO_VCO(streamname, varname, description, classname, objectname)    LOGGING_SCOPED_TIMER_VCO(streamname, varname, description, ::icl_core::logging::eLL_INFO,    classname, objectname)
#ifdef _IC_DEBUG_
# define LOGGING_SCOPED_TIMER_DEBUG_VCO(streamname, varname, description, classname, objectname)   LOGGING_SCOPED_TIMER_VCO(streamname, varname, description, ::icl_core::logging::eLL_DEBUG,   classname, objectname)
# define LOGGING_SCOPED_TIMER_TRACE_VCO(streamname, varname, description, classname, objectname)   LOGGING_SCOPED_TIMER_VCO(streamname, varname, description, ::icl_core::logging::eLL_TRACE,   classname, objectname)
#else
# define LOGGING_SCOPED_TIMER_DEBUG_VCO(streamname, varname, description, classname, objectname) (void)0
# define LOGGING_SCOPED_TIMER_TRACE_VCO(streamname, varname, description, classname, objectname) (void)0
#endif

#endif
