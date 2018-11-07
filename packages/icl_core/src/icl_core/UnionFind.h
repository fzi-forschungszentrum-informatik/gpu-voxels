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
 * \author  Jan Oberlaender <oberlaender@fzi.de>
 * \date    2014-02-12
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_UNION_FIND_H_INCLUDED
#define ICL_CORE_UNION_FIND_H_INCLUDED

#include <vector>
#include <stdexcept>

namespace icl_core {

/*! A very simple union-find data structure for clustering on the set
 *  of indices 0..(n-1).
 */
class UnionFind
{
public:
  UnionFind(const std::size_t n)
    : m_parent(n), m_rank(n, 0)
  {
    for (std::size_t i = 0; i < n; ++i)
    {
      m_parent[i] = i;
    }
  }

  //! Enlarges the data structure by \a n elements.
  void grow(const std::size_t n = 1)
  {
    std::size_t old_size = m_parent.size();
    m_parent.resize(m_parent.size()+n);
    m_rank.resize(m_rank.size()+n, 0);
    for (std::size_t i = old_size; i < m_parent.size(); ++i)
    {
      m_parent[i] = i;
    }
  }

  /*! Not calling this "union" because of the obvious clash with the
   *  C++ keyword.
   */
  void merge(std::size_t x, std::size_t y)
  {
    if (x >= m_parent.size() || y >= m_parent.size())
    {
      throw std::out_of_range("index out of range");
    }
    x = find(x);
    y = find(y);
    if (x == y)
    {
      return;
    }
    if (m_rank[x] < m_rank[y])
    {
      m_parent[x] = y;
    }
    else if (m_rank[x] > m_rank[y])
    {
      m_parent[y] = x;
    }
    else
    {
      m_parent[y] = x;
      ++m_rank[x];
    }
  }

  std::size_t find(std::size_t x)
  {
    if (x >= m_parent.size())
    {
      throw std::out_of_range("index out of range");
    }
    if (m_parent[x] != x)
    {
      m_parent[x] = find(m_parent[x]);
    }
    return m_parent[x];
  }

  std::size_t size() const { return m_parent.size(); }

private:
  std::vector<std::size_t> m_parent;
  std::vector<std::size_t> m_rank;
};

}

#endif
