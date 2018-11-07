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
 * \date    2012-01-24
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_CONFIG_MEMBER_VALUE_IFACE_H_INCLUDED
#define ICL_CORE_CONFIG_MEMBER_VALUE_IFACE_H_INCLUDED

#include "icl_core/TemplateHelper.h"

#include <string>

namespace icl_core {
namespace config {
namespace impl {

/*! Abstract interface for member value configuration readers.
 */
template<typename Q>
class MemberValueIface
{
public:
  virtual ~MemberValueIface() { }

  virtual bool get(std::string const & key,
                   typename icl_core::ConvertToRef<Q>::ToRef value) const = 0;
  virtual std::string getSuffix() const = 0;
  virtual std::string getStringValue() const = 0;
};

}}}

#endif
