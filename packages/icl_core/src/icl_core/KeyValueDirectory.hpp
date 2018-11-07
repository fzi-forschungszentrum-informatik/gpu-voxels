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
 * \date    2008-11-04
 *
 * \brief   Contains KeyValueDirectory
 *
 * Implements a lightweight key/value directory.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_KEY_VALUE_DIRECTORY_HPP_INCLUDED
#define ICL_CORE_KEY_VALUE_DIRECTORY_HPP_INCLUDED

#include "icl_core/KeyValueDirectory.h"

namespace icl_core {

template <typename T>
KeyValueDirectoryIterator<T> KeyValueDirectory<T>::find(const String& query) const
{
  return KeyValueDirectoryIterator<T> (query, this);
}

template <typename T>
bool KeyValueDirectory<T>::get(const String& key, typename ConvertToRef<T>::ToRef value) const
{
  typename KeyValueMap::const_iterator find_it = m_items.find(key);
  if (find_it != m_items.end())
  {
    value = find_it->second;
    return true;
  }
  else
  {
    return false;
  }
}

template <typename T>
bool KeyValueDirectory<T>::hasKey(const String &key) const
{
  typename KeyValueMap::const_iterator find_it = m_items.find(key);
  return find_it != m_items.end();
}

template <typename T>
bool KeyValueDirectory<T>::insert(const String& key, typename ConvertToRef<T>::ToConstRef value)
{
  typename KeyValueMap::const_iterator find_it = m_items.find(key);
  m_items[key] = value;
  return find_it == m_items.end();
}

template <typename T>
KeyValueDirectoryIterator<T>::KeyValueDirectoryIterator(const String& query,
                                                        const KeyValueDirectory<T> *directory)
  : m_directory(directory),
    m_query(query)
{
  reset();
}

template <typename T>
String KeyValueDirectoryIterator<T>::key() const
{
  return m_current_entry->first;
}

template <typename T>
String KeyValueDirectoryIterator<T>::matchGroup(size_t index) const
{
  if (index < m_current_results.size())
  {
    return m_current_results[int(index)];
  }
  else
  {
    return "";
  }
}

template <typename T>
bool KeyValueDirectoryIterator<T>::next()
{
  // If the iterator has been reset (or has just been initialized)
  // we move to the first element.
  if (m_reset == true)
  {
    m_reset = false;
    m_current_entry = m_directory->m_items.begin();
  }
  // Otherwise move to the next iterator position.
  else
  {
    ++m_current_entry;
  }

  // Check if the current iterator position matches the query.
  while (m_current_entry != m_directory->m_items.end() &&
         !::boost::regex_match(m_current_entry->first, m_current_results, m_query))
  {
    ++m_current_entry;
  }

  // Check if there is an element left.
  return m_current_entry != m_directory->m_items.end();
}

template <typename T>
void KeyValueDirectoryIterator<T>::reset()
{
  m_reset = true;
}

template <typename T>
typename ConvertToRef<T>::ToConstRef KeyValueDirectoryIterator<T>::value() const
{
  return m_current_entry->second;
}

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

template <typename T>
ICL_CORE_VC_DEPRECATE_STYLE KeyValueDirectoryIterator<T> KeyValueDirectory<T>::Find(const String& query) const
{
  return find(query);
}

template <typename T>
ICL_CORE_VC_DEPRECATE_STYLE bool KeyValueDirectory<T>::Get(const String& key,
                                                           typename ConvertToRef<T>::ToRef value) const
{
  return get(key, value);
}

template <typename T>
ICL_CORE_VC_DEPRECATE_STYLE bool KeyValueDirectory<T>::HasKey(const String &key) const
{
  return hasKey(key);
}

template <typename T>
ICL_CORE_VC_DEPRECATE_STYLE bool KeyValueDirectory<T>::Insert(const String& key,
                                                              typename ConvertToRef<T>::ToConstRef value)
{
  return insert(key, value);
}

template <typename T>
ICL_CORE_VC_DEPRECATE_STYLE String KeyValueDirectoryIterator<T>::Key() const
{
  return key();
}

template <typename T>
ICL_CORE_VC_DEPRECATE_STYLE String KeyValueDirectoryIterator<T>::MatchGroup(size_t index) const
{
  return matchGroup(index);
}

template <typename T>
ICL_CORE_VC_DEPRECATE_STYLE bool KeyValueDirectoryIterator<T>::Next()
{
  return next();
}

template <typename T>
ICL_CORE_VC_DEPRECATE_STYLE void KeyValueDirectoryIterator<T>::Reset()
{
  reset();
}

template <typename T>
ICL_CORE_VC_DEPRECATE_STYLE typename ConvertToRef<T>::ToConstRef KeyValueDirectoryIterator<T>::Value() const
{
  return value();
}

#endif
/////////////////////////////////////////////////

}

#endif
