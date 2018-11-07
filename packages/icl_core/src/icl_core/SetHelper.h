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
 * \author  Jan Oberländer <oberlaen@fzi.de>
 * \date    2010-08-07
 *
 * \brief   Contains helper functions to work on STL sets.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_SET_HELPER_H_INCLUDED
#define ICL_CORE_SET_HELPER_H_INCLUDED

#include <set>

namespace icl_core {

//! Helper enum to describe relations between sets.
enum SetRelation
{
  eSR_EQUAL,
  eSR_PROPER_SUPERSET,
  eSR_PROPER_SUBSET,
  eSR_INTERSECTION_NONEMPTY,
  eSR_DISJUNCT
};

typedef ICL_CORE_VC_DEPRECATE SetRelation tSetRelation ICL_CORE_GCC_DEPRECATE;

/*! Generic helper function to determine the relationships between two sets.
 */
template <typename T, typename TCompare>
SetRelation setRelation(const std::set<T, TCompare>& s1, const std::set<T, TCompare>& s2)
{
  bool equal = true;
  bool super = true;
  bool sub = true;
  bool inter = false;
  TCompare compare;
  typename std::set<T, TCompare>::const_iterator i1 = s1.begin(), i2 = s2.begin();
  while (i1 != s1.end() && i2 != s2.end())
  {
    if ((*i1) == (*i2))
    {
      inter = true; // Intersection found
      ++i1, ++i2;
    }
    else
    {
      equal = false; // Unequal elements found, cannot be equal
      if (compare(*i1, *i2))
      {
        sub = false; // *it is definitely not in other set, so we're no subset
        ++i1;
      }
      else
      {
        super = false; // *other_it is definitely not in this set, so we're no superset
        ++i2;
      }
    }
  }
  if (i1 != s1.end())
  {
    equal = false;
    sub = false; // *it is definitely not in other set, so we're no subset
  }
  if (i2 != s2.end())
  {
    equal = false;
    super = false; // *other_it is definitely not in this set, so we're no superset
  }
  if (equal) { return eSR_EQUAL; }
  if (super) { return eSR_PROPER_SUPERSET; }
  if (sub) { return eSR_PROPER_SUBSET; }
  if (inter) { return eSR_INTERSECTION_NONEMPTY; }
  return eSR_DISJUNCT;
}

}

#endif
