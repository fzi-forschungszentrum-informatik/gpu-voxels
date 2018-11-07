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
 * \date    2015-01-15
 *
 */
//----------------------------------------------------------------------
#include <icl_core/UnionFind.h>

#include <boost/test/unit_test.hpp>

using icl_core::UnionFind;

BOOST_AUTO_TEST_SUITE(ts_UnionFind)

BOOST_AUTO_TEST_CASE(UnionFindTest)
{
  UnionFind uf(5);
  BOOST_CHECK_EQUAL(uf.size(), 5);
  BOOST_CHECK_EQUAL(uf.find(0), 0);
  BOOST_CHECK_EQUAL(uf.find(1), 1);
  BOOST_CHECK_EQUAL(uf.find(2), 2);
  BOOST_CHECK_EQUAL(uf.find(3), 3);
  BOOST_CHECK_EQUAL(uf.find(4), 4);

  uf.merge(0, 1);
  uf.merge(2, 3);
  BOOST_CHECK_EQUAL(uf.find(0), 0);
  BOOST_CHECK_EQUAL(uf.find(1), 0);
  BOOST_CHECK_EQUAL(uf.find(2), 2);
  BOOST_CHECK_EQUAL(uf.find(3), 2);
  BOOST_CHECK_EQUAL(uf.find(4), 4);

  uf.merge(2, 4);
  BOOST_CHECK_EQUAL(uf.find(0), 0);
  BOOST_CHECK_EQUAL(uf.find(1), 0);
  BOOST_CHECK_EQUAL(uf.find(2), 2);
  BOOST_CHECK_EQUAL(uf.find(3), 2);
  BOOST_CHECK_EQUAL(uf.find(4), 2);

  // Copying the structure and merging in different directions
  {
    UnionFind uf_forward = uf;
    UnionFind uf_backward = uf;
    uf_forward.merge(0, 3);
    BOOST_CHECK_EQUAL(uf_forward.find(0), 0);
    BOOST_CHECK_EQUAL(uf_forward.find(1), 0);
    BOOST_CHECK_EQUAL(uf_forward.find(2), 0);
    BOOST_CHECK_EQUAL(uf_forward.find(3), 0);
    BOOST_CHECK_EQUAL(uf_forward.find(4), 0);

    uf_backward.merge(3, 0);
    BOOST_CHECK_EQUAL(uf_backward.find(0), 2);
    BOOST_CHECK_EQUAL(uf_backward.find(1), 2);
    BOOST_CHECK_EQUAL(uf_backward.find(2), 2);
    BOOST_CHECK_EQUAL(uf_backward.find(3), 2);
    BOOST_CHECK_EQUAL(uf_backward.find(4), 2);
  }

  // Enlarging the structure
  uf.grow(2);
  BOOST_CHECK_EQUAL(uf.size(), 7);
  BOOST_CHECK_EQUAL(uf.find(0), 0);
  BOOST_CHECK_EQUAL(uf.find(1), 0);
  BOOST_CHECK_EQUAL(uf.find(2), 2);
  BOOST_CHECK_EQUAL(uf.find(3), 2);
  BOOST_CHECK_EQUAL(uf.find(4), 2);
  BOOST_CHECK_EQUAL(uf.find(5), 5);
  BOOST_CHECK_EQUAL(uf.find(6), 6);

  // Merging on the enlarged structure
  uf.merge(5, 6);
  uf.merge(4, 5);
  BOOST_CHECK_EQUAL(uf.find(0), 0);
  BOOST_CHECK_EQUAL(uf.find(1), 0);
  BOOST_CHECK_EQUAL(uf.find(2), 2);
  BOOST_CHECK_EQUAL(uf.find(3), 2);
  BOOST_CHECK_EQUAL(uf.find(4), 2);
  BOOST_CHECK_EQUAL(uf.find(5), 2);
  BOOST_CHECK_EQUAL(uf.find(6), 2);

  uf.merge(1, 6);
  BOOST_CHECK_EQUAL(uf.find(0), 2);
  BOOST_CHECK_EQUAL(uf.find(1), 2);
  BOOST_CHECK_EQUAL(uf.find(2), 2);
  BOOST_CHECK_EQUAL(uf.find(3), 2);
  BOOST_CHECK_EQUAL(uf.find(4), 2);
  BOOST_CHECK_EQUAL(uf.find(5), 2);
  BOOST_CHECK_EQUAL(uf.find(6), 2);
}

BOOST_AUTO_TEST_SUITE_END()
