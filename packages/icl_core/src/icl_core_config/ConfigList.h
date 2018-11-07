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
#ifndef ICL_CORE_CONFIG_CONFIG_LIST_H_INCLUDED
#define ICL_CORE_CONFIG_CONFIG_LIST_H_INCLUDED

#include "icl_core_config/ConfigHelper.h"
#include "icl_core_config/ConfigValues.h"
#include "icl_core_config/MemberEnum.h"
#include "icl_core_config/MemberValue.h"

#include <iterator>
#include <list>
#include <boost/assign/list_of.hpp>
#include <boost/regex.hpp>

#define CONFIG_LIST(cls, prefix, members, result)                      \
  (new icl_core::config::ConfigList<cls, ICL_CORE_CONFIG_TYPEOF(result)>(prefix, members, result))

#define MEMBER_MAPPING(cls, arg) boost::assign::list_of<icl_core::config::impl::MemberValueIface<cls>*> arg


namespace icl_core {
namespace config {

/*! Reads a list of equally structured configuration entries.
 *
 *  The target can be either a list of single values or a
 *  list containing structs.
 *
 *  The output iterator is pre-configured to append to a
 *  std::list<T>. If you want it to write to a different container
 *  or use a pre-allocated container you have to specify the
 *  \a OutputIterator template parameter.
 */
template<typename T, class OutputIterator = std::back_insert_iterator<std::list<T> > >
class ConfigList : public impl::ConfigValueIface
{
public:
  ConfigList(std::string config_prefix,
             std::list<impl::MemberValueIface<T>*> members,
             OutputIterator result)
    : m_config_prefix(config_prefix),
      m_members(members),
      m_result(result)
  {
  }
  virtual ~ConfigList() {}

  virtual bool get(std::string const & prefix, icl_core::logging::LogStream& log_stream) const
  {
    bool result = false;
    bool error = false;
    ConfigIterator cIter = ConfigManager::instance().find(
      boost::regex_replace(prefix + m_config_prefix, boost::regex("\\/"), "\\\\\\/") + "\\/(.+?)(\\/(.+))?");
    while(cIter.next())
    {
      bool found = false;
      std::string element = cIter.matchGroup(1);
      std::string suffix = cIter.matchGroup(3);
      for (typename std::list<impl::MemberValueIface<T>*>::const_iterator it = m_members.begin();
           !found && it != m_members.end(); ++it)
      {
        if (suffix == (*it)->getSuffix())
        {
          found = true;
          if ((*it)->get(cIter.matchGroup(0), m_element_map[element]))
          {
            m_string_value_map[element] += " " + suffix + "=" + (*it)->getStringValue();
            
            SLOGGING_TRACE_C(log_stream, ConfigList,
                            "Read parameter " << cIter.matchGroup(0) << " = "
                            << (*it)->getStringValue() << icl_core::logging::endl);
            result = true;
          }
          else
          {
            SLOGGING_ERROR_C(log_stream, ConfigList,
                           "Error reading configuration parameter \"" << cIter.matchGroup(0) << " = "
                           << (*it)->getStringValue() << icl_core::logging::endl);
            error = true;
          }
        }
      }
    }

    if (error)
    {
      result = false;
    }

    if (result == true)
    {
      for (typename std::map<std::string, T>::const_iterator it = m_element_map.begin();
           it != m_element_map.end(); ++it)
      {
        *m_result = it->second;
        ++m_result;
      }

      m_string_value = m_config_prefix + " = [";
      for (std::map<std::string, std::string>::const_iterator it = m_string_value_map.begin();
           it != m_string_value_map.end(); ++it)
      {
        m_string_value += " " + it->first + ": (" + it->second + ")";
      }
      m_string_value += " ]";
    }

    return result;
  }

  virtual icl_core::String key() const { return m_config_prefix; }
  virtual icl_core::String stringValue() const { return m_string_value; }

private:
  std::string m_config_prefix;
  std::list<impl::MemberValueIface<T>*> m_members;
  mutable std::map<std::string, T> m_element_map;
  mutable std::map<std::string, std::string> m_string_value_map;
  mutable OutputIterator m_result;
  mutable std::string m_string_value;
};

}}

#endif
