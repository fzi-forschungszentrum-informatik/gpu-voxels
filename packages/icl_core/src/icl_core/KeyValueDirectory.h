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
#ifndef ICL_CORE_KEY_VALUE_DIRECTORY_H_INCLUDED
#define ICL_CORE_KEY_VALUE_DIRECTORY_H_INCLUDED

#include <boost/regex.hpp>

#include "icl_core/BaseTypes.h"
#include "icl_core/Map.h"
#include "icl_core/TemplateHelper.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {

template <typename T>
class KeyValueDirectoryIterator;

/*! Implements a lightweight key/value directory.
 */
template <typename T>
class KeyValueDirectory
{
  friend class KeyValueDirectoryIterator<T>;
public:

  /*! Finds all entries which match the specified \a query.  Boost
   *  regular expressions are allowed for the query.
   *
   *  \returns An iterator which iterates over all entries, which
   *           match the specified \a query.
   */
  KeyValueDirectoryIterator<T> find(const String& query) const;

  /*! Get a \a value for the specified \a key.
   *
   *  \returns \c true if a configuration value for the key exists, \c
   *           false otherwise.
   */
  bool get(const String& key, typename ConvertToRef<T>::ToRef value) const;

  /*! Check if the \a key exists.
   */
  bool hasKey(const String& key) const;

  /*! Insert a new \a key / \a value pair.
   *
   *  \returns \c true if a new element was inserted, \c false if an
   *           existing element was replaced.
   */
  bool insert(const String& key, typename ConvertToRef<T>::ToConstRef value);

  ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  /*! Finds all entries which match the specified \a query.
   * \deprecated Obsolete coding style.
   */
  KeyValueDirectoryIterator<T> ICL_CORE_VC_DEPRECATE_STYLE
  Find(const String& query) const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Get a \a value for the specified \a key.
   * \deprecated Obsolete coding style.
   */
  bool ICL_CORE_VC_DEPRECATE_STYLE
  Get(const String& key, typename ConvertToRef<T>::ToRef value) const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Check if the \a key exists.
   *  \deprecated Obsolete coding style.
   */
  bool ICL_CORE_VC_DEPRECATE_STYLE HasKey(const String& key) const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Insert a new \a key / \a value pair.
   * \deprecated Obsolete coding style.
   */
  bool ICL_CORE_VC_DEPRECATE_STYLE
  Insert(const String& key, typename ConvertToRef<T>::ToConstRef value) ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
  /////////////////////////////////////////////////

private:
  typedef Map<String, T> KeyValueMap;
  KeyValueMap m_items;
};

/*!
 * Implements an iterator for regular expression querys to
 * a key/value directory.
 */
template <typename T>
class KeyValueDirectoryIterator
{
public:
  /*!
   * Create a new iterator for the \a query on the \a directory.
   */
  KeyValueDirectoryIterator(const String& query, const KeyValueDirectory<T> *directory);

  /*!
   * Get the key of the current match result.
   */
  String key() const;

  /*!
   * Get the match group at the specified \a index.
   * \n
   * Remark: Match groups are the equivalent of Perl's (or sed's)
   * $n references.
   */
  String matchGroup(size_t index) const;

  /*!
   * Move to the next query result.
   *
   * \returns \a false if no next query result exists.
   */
  bool next();

  /*!
   * Resets the iterator. You have to call Next() to move it
   * to the first matching configuration entry.
   */
  void reset();

  /*!
   * Get the value of the current match result.
   */
  typename ConvertToRef<T>::ToConstRef value() const;

  ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_
  /*!
   * Get the key of the current match result.
   * \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE String Key() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*!
   * Get the match group at the specified \a index.
   * \n
   * Remark: Match groups are the equivalent of Perl's (or sed's)
   * $n references.
   * \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE String MatchGroup(size_t index) const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*!
   * Move to the next query result.
   *
   * \returns \a false if no next query result exists.
   * \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool Next() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*!
   * Resets the iterator. You have to call Next() to move it
   * to the first matching configuration entry.
   * \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void Reset() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*!
   * Get the value of the current match result.
   * \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE typename ConvertToRef<T>::ToConstRef Value() const ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
  /////////////////////////////////////////////////

private:
  const KeyValueDirectory<T> *m_directory;
  boost::regex m_query;
  boost::match_results<icl_core::String::const_iterator> m_current_results;

  typename KeyValueDirectory<T>::KeyValueMap::const_iterator m_current_entry;
  bool m_reset;
};

}

#include "icl_core/KeyValueDirectory.hpp"

#endif
