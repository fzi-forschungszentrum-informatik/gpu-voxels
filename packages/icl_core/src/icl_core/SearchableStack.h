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
 * \date    2010-07-16
 *
 * \brief   Contains icl_core::SearchableStack
 *
 * \b icl_core::SearchableStack
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_SEARCHABLE_STACK_H_INCLUDED
#define ICL_CORE_SEARCHABLE_STACK_H_INCLUDED

#include <functional>
#include <vector>

#include "icl_core/BaseTypes.h"
#include "icl_core/TemplateHelper.h"

namespace icl_core {

/*! A stack implementation that allows const-iteration over the
 *  stored elements and searching for a specific element.  Apart
 *  from these additional features it behaves like std::stack.
 */
template <typename T, typename TCompare = std::equal_to<T>, typename TAlloc = std::allocator<T> >
class SearchableStack : protected std::vector<T, TAlloc>
{
public:
  /*! Const iterator type.  Non-const iterators are not provided,
   *  only the top element may be modified.
   */
  typedef typename std::vector<T>::const_iterator const_iterator;
  //! Const reverse iterator type.
  typedef typename std::vector<T>::const_reverse_iterator const_reverse_iterator;
  //! Size type.
  typedef typename std::vector<T>::size_type size_type;

  //! Constructs an empty stack.
  SearchableStack() : std::vector<T, TAlloc>(), m_comp() { }

  //! Clears the stack.
  void clear() { std::vector<T, TAlloc>::clear(); }
  //! Returns \c true if the stack is empty.
  bool empty() const { return std::vector<T, TAlloc>::empty(); }
  //! Returns the number of elements on the stack.
  size_type size() const { return std::vector<T, TAlloc>::size(); }

  //! Pushes an element onto the stack.
  void push(typename ConvertToRef<T>::ToConstRef t) { std::vector<T, TAlloc>::push_back(t); }
  //! Pops an element off the stack.
  void pop() { std::vector<T, TAlloc>::pop_back(); }
  //! Returns a reference to the top stack element.
  typename ConvertToRef<T>::ToRef top() { return std::vector<T, TAlloc>::back(); }
  //! Returns a const reference to the top stack element.
  typename ConvertToRef<T>::ToConstRef top() const { return std::vector<T, TAlloc>::back(); }

  //! Returns an iterator to the bottom element of the stack.
  const_iterator begin() const { return std::vector<T, TAlloc>::begin(); }
  //! Returns an iterator to the top end of the stack.
  const_iterator end() const { return std::vector<T, TAlloc>::end(); }
  //! Returns a reverse iterator to the top element of the stack.
  const_reverse_iterator rbegin() const { return std::vector<T, TAlloc>::rbegin(); }
  //! Returns a reverse iterator to the bottom end of the stack.
  const_reverse_iterator rend() const { return std::vector<T, TAlloc>::rend(); }

  //! Finds a specific element on the stack by linear search.
  const_iterator find(typename ConvertToRef<T>::ToConstRef t) const
  {
    for (const_iterator it = begin(); it != end(); ++it)
    {
      if (m_comp(*it, t)) { return it; }
    }
    return end();
  }

  TCompare m_comp;
};

typedef SearchableStack<uint8_t> Unsigned8SearchableStack;
typedef SearchableStack<uint16_t> Unsigned16SearchableStack;
typedef SearchableStack<uint32_t> Unsigned32SearchableStack;
typedef SearchableStack<uint64_t> Unsigned64SearchableStack;
typedef SearchableStack<int8_t> Signed8SearchableStack;
typedef SearchableStack<int16_t> Signed16SearchableStack;
typedef SearchableStack<int32_t> Signed32SearchableStack;
typedef SearchableStack<int64_t> Signed64SearchableStack;
typedef SearchableStack<float> FloaSearchableStack;
typedef SearchableStack<double> DoubleSearchableStack;

}

#endif
