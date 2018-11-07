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
#ifndef ICL_CORE_CONFIG_MEMBER_ENUM_H_INCLUDED
#define ICL_CORE_CONFIG_MEMBER_ENUM_H_INCLUDED

#include "icl_core/RemoveMemberPointer.h"
#include "icl_core/TemplateHelper.h"
#include "icl_core_config/ConfigHelper.h"
#include "icl_core_config/ConfigManager.h"
#include "icl_core_config/MemberValueIface.h"

#include <algorithm>
#include <iterator>
#include <string>
#include <vector>
#include <boost/function.hpp>
#include <boost/lambda/bind.hpp>

#define MEMBER_ENUM_1(suffix, cls, member1, descriptions)                                      \
  (new icl_core::config::MemberEnum<                                                           \
     icl_core::RemoveMemberPointer<ICL_CORE_CONFIG_TYPEOF(&cls::member1)>::Type, cls>(         \
     suffix,                                                                                   \
     boost::lambda::bind(&cls::member1, boost::lambda::_1), descriptions))

#define MEMBER_ENUM_2(suffix, cls, member1, member2, descriptions)                             \
  (new icl_core::config::MemberEnum<                                                           \
     icl_core::RemoveMemberPointer<ICL_CORE_CONFIG_TYPEOF(                                     \
       &icl_core::RemoveMemberPointer<ICL_CORE_CONFIG_TYPEOF(                                  \
         &cls::member1)>::Type::member2)>::Type, cls>(                                         \
     suffix,                                                                                   \
     boost::lambda::bind(                                                                      \
       &icl_core::RemoveMemberPointer<ICL_CORE_CONFIG_TYPEOF(&cls::member1)>::Type::member2,   \
       boost::lambda::bind(&cls::member1, boost::lambda::_1)), descriptions))

#define MEMBER_ENUM_3(suffix, cls, member1, member2, member3, descriptions)                    \
  (new icl_core::config::MemberEnum<                                                           \
     icl_core::RemoveMemberPointer<ICL_CORE_CONFIG_TYPEOF(                                     \
       &icl_core::RemoveMemberPointer<ICL_CORE_CONFIG_TYPEOF(                                  \
         &icl_core::RemoveMemberPointer<ICL_CORE_CONFIG_TYPEOF(                                \
           &cls::member1)>::Type::member2)>::Type::member3)>::Type, cls>(                      \
     suffix,                                                                                   \
     boost::lambda::bind(                                                                      \
       &icl_core::RemoveMemberPointer<ICL_CORE_CONFIG_TYPEOF(                                  \
         &icl_core::RemoveMemberPointer<ICL_CORE_CONFIG_TYPEOF(                                \
           &cls::member1)>::Type::member2)>::Type::member3,                                    \
       boost::lambda::bind(                                                                    \
         &icl_core::RemoveMemberPointer<ICL_CORE_CONFIG_TYPEOF(&cls::member1)>::Type::member2, \
         boost::lambda::bind(&cls::member1, boost::lambda::_1))), descriptions))

namespace icl_core {
namespace config {

template<typename T, typename Q, typename V = int32_t>
class MemberEnum : public impl::MemberValueIface<Q>
{
public:
  MemberEnum(std::string const & config_suffix,
             boost::function<T&(Q&)> accessor,
             char const * const * descriptions,
             char const * end_marker = NULL)
    : m_config_suffix(config_suffix),
      m_accessor(accessor)
  {
    if (descriptions != NULL)
    {
      for (size_t i = 0;
           ((end_marker == NULL) && (descriptions[i] != NULL)) ||
            ((end_marker != NULL) && (::strcmp(descriptions[i], end_marker) != 0));
           ++i)
      {
        m_descriptions.push_back(std::string(descriptions[i]));
      }
    }
  }
  MemberEnum(std::string const & config_suffix,
             boost::function<T&(Q&)> accessor,
             std::vector<std::string> const & descriptions)
    : m_config_suffix(config_suffix),
      m_accessor(accessor)
  {
    std::copy(descriptions.begin(), descriptions.end(), std::back_inserter(m_descriptions));
  }
  virtual ~MemberEnum() {}

  virtual bool get(std::string const & key,
                   typename icl_core::ConvertToRef<Q>::ToRef value) const
  {
    bool result = false;
    if (ConfigManager::instance().get(key, m_str_value))
    {
      V raw_value;
      if (icl_core::string2Enum(m_str_value, raw_value, m_descriptions))
      {
        m_accessor(value) = T(raw_value);
        result = true;
      }
      else
      {
        result = false;
      }
    }
    else
    {
      result = false;
    }
    return result;
  }

  virtual std::string getSuffix() const { return m_config_suffix; }
  virtual std::string getStringValue() const { return m_str_value; }

private:
  std::string m_config_suffix;
  boost::function<T&(Q&)> m_accessor;
  std::vector<std::string> m_descriptions;
  mutable std::string m_str_value;
};

}}

#endif
