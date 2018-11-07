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
 * \brief   Contains icl_core::tSearchableStack
 *
 * \b icl_core::tSearchableStack
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_T_SEARCHABLE_STACK_H_INCLUDED
#define ICL_CORE_T_SEARCHABLE_STACK_H_INCLUDED

#include <functional>
#include <vector>

#include "icl_core/BaseTypes.h"
#include "icl_core/Deprecate.h"
#include "icl_core/TemplateHelper.h"

namespace icl_core {

/*! A stack implementation that allows const-iteration over the
 *  stored elements and searching for a specific element.  Apart
 *  from these additional features it behaves like std::stack.
 */
template <typename T, typename Compare = std::equal_to<T>, typename Alloc = std::allocator<T> >
class ICL_CORE_VC_DEPRECATE tSearchableStack : protected std::vector<T, Alloc>
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
  tSearchableStack() : std::vector<T, Alloc>(), m_comp() { }

  //! Clears the stack.
  void clear() { std::vector<T, Alloc>::clear(); }
  //! Returns \c true if the stack is empty.
  bool empty() const { return std::vector<T, Alloc>::empty(); }
  //! Returns the number of elements on the stack.
  size_type size() const { return std::vector<T, Alloc>::size(); }

  //! Pushes an element onto the stack.
  void push(typename ConvertToRef<T>::ToConstRef t) { std::vector<T, Alloc>::push_back(t); }
  //! Pops an element off the stack.
  void pop() { std::vector<T, Alloc>::pop_back(); }
  //! Returns a reference to the top stack element.
  typename ConvertToRef<T>::ToRef top() { return std::vector<T, Alloc>::back(); }
  //! Returns a const reference to the top stack element.
  typename ConvertToRef<T>::ToConstRef top() const { return std::vector<T, Alloc>::back(); }

  //! Returns an iterator to the bottom element of the stack.
  const_iterator begin() const { return std::vector<T, Alloc>::begin(); }
  //! Returns an iterator to the top end of the stack.
  const_iterator end() const { return std::vector<T, Alloc>::end(); }
  //! Returns a reverse iterator to the top element of the stack.
  const_reverse_iterator rbegin() const { return std::vector<T, Alloc>::rbegin(); }
  //! Returns a reverse iterator to the bottom end of the stack.
  const_reverse_iterator rend() const { return std::vector<T, Alloc>::rend(); }

  //! Finds a specific element on the stack by linear search.
  const_iterator find(typename ConvertToRef<T>::ToConstRef t) const
  {
    for (const_iterator it = begin(); it != end(); ++it)
    {
      if (m_comp(*it, t)) { return it; }
    }
    return end();
  }

  Compare m_comp;
} ICL_CORE_GCC_DEPRECATE;

typedef tSearchableStack<tUnsigned8> tUnsigned8SearchableStack;
typedef tSearchableStack<tUnsigned16> tUnsigned16SearchableStack;
typedef tSearchableStack<tUnsigned32> tUnsigned32SearchableStack;
typedef tSearchableStack<tUnsigned64> tUnsigned64SearchableStack;
typedef tSearchableStack<tSigned8> tSigned8SearchableStack;
typedef tSearchableStack<tSigned16> tSigned16SearchableStack;
typedef tSearchableStack<tSigned32> tSigned32SearchableStack;
typedef tSearchableStack<tSigned64> tSigned64SearchableStack;
typedef tSearchableStack<tFloat> tFloatSearchableStack;
typedef tSearchableStack<tDouble> tDoubleSearchableStack;

}

#endif
