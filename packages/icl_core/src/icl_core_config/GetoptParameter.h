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
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2009-03-12
 *
 * \brief   Contains GetoptParameter
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_CONFIG_GETOPT_PARAMETER_H_INCLUDED
#define ICL_CORE_CONFIG_GETOPT_PARAMETER_H_INCLUDED

#include "icl_core/BaseTypes.h"
#include "icl_core/Vector.h"
#include "icl_core_config/ImportExport.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {
namespace config {

class ICL_CORE_CONFIG_IMPORT_EXPORT GetoptParameter
{
public:
  /*! Create a new commandline parameter.
   *
   *  \param option The long option name of this parameter. If \a
   *         option ends with a colon (":") then the parameter also
   *         expects a value.
   *  \param short_option The short option name of this parameter.  If
   *         this is set to the empty string then no short option is
   *         used.
   *  \param help The help text for this parameter.
   *  \param is_prefix Set to \c true if this is a prefix option.
   *         Prefix Options are options like "-o/asd/asd".
   *
   *  \see GetoptParameter for details about the syntax of the \a
   *  option parameter.
   */
  GetoptParameter(const icl_core::String& option, const icl_core::String& short_option,
                  const icl_core::String& help, bool is_prefix = false);

  //! Get the long option name.
  icl_core::String option() const { return m_option; }
  //! Get the short option name.
  icl_core::String shortOption() const { return m_short_option; }
  //! Check if the option also expects a value.
  bool hasValue() const { return m_has_value; }
  //! Get the help text.
  icl_core::String help() const { return m_help; }

  //! Check if this is a prefix option.
  bool isPrefixOption() const { return m_is_prefix; }

  ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  /*! Get the long option name.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE icl_core::String Option() const ICL_CORE_GCC_DEPRECATE_STYLE;
  /*! Get the short option name.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE icl_core::String ShortOption() const ICL_CORE_GCC_DEPRECATE_STYLE;
  /*! Check if the option also expects a value.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool HasValue() const ICL_CORE_GCC_DEPRECATE_STYLE;
  /*! Get the help text.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE icl_core::String Help() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Check if this is a prefix option.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool IsPrefixOption() const ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
  /////////////////////////////////////////////////

private:
  icl_core::String m_option;
  icl_core::String m_short_option;
  icl_core::String m_help;
  bool m_has_value;
  bool m_is_prefix;
};

typedef icl_core::Vector<GetoptParameter> GetoptParameterList;

}
}

#endif
