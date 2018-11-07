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
 * \author  Florian Kuhnt <kuhnt@fzi.de>
 * \date    2014-05-07
 *
 * \brief   Contains GetoptPositionalParameter
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_CONFIG_GETOPT_POSITIONAL_PARAMETER_H_INCLUDED
#define ICL_CORE_CONFIG_GETOPT_POSITIONAL_PARAMETER_H_INCLUDED

#include "icl_core/BaseTypes.h"
#include "icl_core/Vector.h"
#include "icl_core_config/ImportExport.h"

namespace icl_core {
namespace config {

class ICL_CORE_CONFIG_IMPORT_EXPORT GetoptPositionalParameter
{
public:
  /*! Create a new positional commandline parameter.
   *
   * \param name The name of the parameter.
   * \param help A help text that will be used in the generic help.
   * \param is_optional Iff the parameter is an optional parameter.
   */
  GetoptPositionalParameter(const icl_core::String& name,
                            const icl_core::String& help,
                            const bool is_optional=false);

  //! Get the option name.
  icl_core::String name() const { return m_name; }
  //! Get the help text.
  icl_core::String help() const { return m_help; }
  //! Get if the parameter is optional.
  bool isOptional() const { return m_is_optional; }

private:
  icl_core::String m_name;
  icl_core::String m_help;
  bool m_is_optional;
};

typedef icl_core::Vector<GetoptPositionalParameter> GetoptPositionalParameterList;

}
}

#endif
