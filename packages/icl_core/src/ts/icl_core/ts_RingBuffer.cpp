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
 * \date    2011-11-07
 *
 */
//----------------------------------------------------------------------
#include <icl_core/RingBuffer.h>

#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>

using icl_core::RingBuffer;

BOOST_AUTO_TEST_SUITE(ts_RingBuffer)

typedef boost::mpl::list<char, short, int, long, float, double> TestTypes;

BOOST_AUTO_TEST_CASE_TEMPLATE(RingBufferCapacity, T, TestTypes)
{
  RingBuffer<T> ringbuffer(10);

  // Check empty
  BOOST_CHECK_EQUAL(ringbuffer.size(), 0u);
  BOOST_CHECK_EQUAL(ringbuffer.capacity(), 10u);
  BOOST_CHECK_EQUAL(ringbuffer.reserve(), 10u);
  BOOST_CHECK_EQUAL(ringbuffer.empty(), true);
  BOOST_CHECK_EQUAL(ringbuffer.full(), false);

  // Check some exceptions
  bool read_exception_called = false;
  try { ringbuffer.read(); }
  catch (std::out_of_range&) { read_exception_called = true; }
  BOOST_CHECK_EQUAL(read_exception_called, true);

  // Add one element and check size
  ringbuffer.write(0);

  BOOST_CHECK_EQUAL(ringbuffer.size(), 1u);
  BOOST_CHECK_EQUAL(ringbuffer.capacity(), 10u);
  BOOST_CHECK_EQUAL(ringbuffer.reserve(), 9u);
  BOOST_CHECK_EQUAL(ringbuffer.empty(), false);
  BOOST_CHECK_EQUAL(ringbuffer.full(), false);

  // Add six more elements, check again
  ringbuffer.write(1);
  ringbuffer.write(2);
  ringbuffer.write(3);
  ringbuffer.write(4);
  ringbuffer.write(5);
  ringbuffer.write(6);

  BOOST_CHECK_EQUAL(ringbuffer.size(), 7u);
  BOOST_CHECK_EQUAL(ringbuffer.capacity(), 10u);
  BOOST_CHECK_EQUAL(ringbuffer.reserve(), 3u);
  BOOST_CHECK_EQUAL(ringbuffer.empty(), false);
  BOOST_CHECK_EQUAL(ringbuffer.full(), false);

  // Fill up, check full
  ringbuffer.write(7);
  ringbuffer.write(8);
  ringbuffer.write(9);

  BOOST_CHECK_EQUAL(ringbuffer.size(), 10u);
  BOOST_CHECK_EQUAL(ringbuffer.capacity(), 10u);
  BOOST_CHECK_EQUAL(ringbuffer.reserve(), 0u);
  BOOST_CHECK_EQUAL(ringbuffer.empty(), false);
  BOOST_CHECK_EQUAL(ringbuffer.full(), true);

  // Check all contained element values
  BOOST_CHECK_EQUAL(ringbuffer.at(0), T(0));
  BOOST_CHECK_EQUAL(ringbuffer.at(1), T(1));
  BOOST_CHECK_EQUAL(ringbuffer.at(2), T(2));
  BOOST_CHECK_EQUAL(ringbuffer.at(3), T(3));
  BOOST_CHECK_EQUAL(ringbuffer.at(4), T(4));
  BOOST_CHECK_EQUAL(ringbuffer.at(5), T(5));
  BOOST_CHECK_EQUAL(ringbuffer.at(6), T(6));
  BOOST_CHECK_EQUAL(ringbuffer.at(7), T(7));
  BOOST_CHECK_EQUAL(ringbuffer.at(8), T(8));
  BOOST_CHECK_EQUAL(ringbuffer.at(9), T(9));

  // Remove the first four elements
  BOOST_CHECK_EQUAL(ringbuffer.read(), T(0));
  BOOST_CHECK_EQUAL(ringbuffer.read(), T(1));
  BOOST_CHECK_EQUAL(ringbuffer.read(), T(2));
  BOOST_CHECK_EQUAL(ringbuffer.read(), T(3));

  // Check size
  BOOST_CHECK_EQUAL(ringbuffer.size(), 6u);
  BOOST_CHECK_EQUAL(ringbuffer.capacity(), 10u);
  BOOST_CHECK_EQUAL(ringbuffer.reserve(), 4u);
  BOOST_CHECK_EQUAL(ringbuffer.empty(), false);
  BOOST_CHECK_EQUAL(ringbuffer.full(), false);

  // Check remaining element values
  BOOST_CHECK_EQUAL(ringbuffer.at(0), T(4));
  BOOST_CHECK_EQUAL(ringbuffer.at(1), T(5));
  BOOST_CHECK_EQUAL(ringbuffer.at(2), T(6));
  BOOST_CHECK_EQUAL(ringbuffer.at(3), T(7));
  BOOST_CHECK_EQUAL(ringbuffer.at(4), T(8));
  BOOST_CHECK_EQUAL(ringbuffer.at(5), T(9));

  // Fill up again, check full
  ringbuffer.write(10);
  ringbuffer.write(11);
  ringbuffer.write(12);
  ringbuffer.write(13);

  BOOST_CHECK_EQUAL(ringbuffer.size(), 10u);
  BOOST_CHECK_EQUAL(ringbuffer.capacity(), 10u);
  BOOST_CHECK_EQUAL(ringbuffer.reserve(), 0u);
  BOOST_CHECK_EQUAL(ringbuffer.empty(), false);
  BOOST_CHECK_EQUAL(ringbuffer.full(), true);

  // Check all element values
  BOOST_CHECK_EQUAL(ringbuffer.at(0), T(4));
  BOOST_CHECK_EQUAL(ringbuffer.at(1), T(5));
  BOOST_CHECK_EQUAL(ringbuffer.at(2), T(6));
  BOOST_CHECK_EQUAL(ringbuffer.at(3), T(7));
  BOOST_CHECK_EQUAL(ringbuffer.at(4), T(8));
  BOOST_CHECK_EQUAL(ringbuffer.at(5), T(9));
  BOOST_CHECK_EQUAL(ringbuffer.at(6), T(10));
  BOOST_CHECK_EQUAL(ringbuffer.at(7), T(11));
  BOOST_CHECK_EQUAL(ringbuffer.at(8), T(12));
  BOOST_CHECK_EQUAL(ringbuffer.at(9), T(13));

  // Check some exceptions
  bool write_exception_called = false;
  try { ringbuffer.write(14); }
  catch (std::out_of_range&) { write_exception_called = true; }
  BOOST_CHECK_EQUAL(write_exception_called, true);

  // Skip some elements, check size
  ringbuffer.skip();
  ringbuffer.skip(3);

  BOOST_CHECK_EQUAL(ringbuffer.size(), 6u);
  BOOST_CHECK_EQUAL(ringbuffer.capacity(), 10u);
  BOOST_CHECK_EQUAL(ringbuffer.reserve(), 4u);
  BOOST_CHECK_EQUAL(ringbuffer.empty(), false);
  BOOST_CHECK_EQUAL(ringbuffer.full(), false);

  // Check all element values
  BOOST_CHECK_EQUAL(ringbuffer.at(0), T(8));
  BOOST_CHECK_EQUAL(ringbuffer.at(1), T(9));
  BOOST_CHECK_EQUAL(ringbuffer.at(2), T(10));
  BOOST_CHECK_EQUAL(ringbuffer.at(3), T(11));
  BOOST_CHECK_EQUAL(ringbuffer.at(4), T(12));
  BOOST_CHECK_EQUAL(ringbuffer.at(5), T(13));

  // Empty buffer, check empty
  ringbuffer.skip(10);

  BOOST_CHECK_EQUAL(ringbuffer.size(), 0u);
  BOOST_CHECK_EQUAL(ringbuffer.capacity(), 10u);
  BOOST_CHECK_EQUAL(ringbuffer.reserve(), 10u);
  BOOST_CHECK_EQUAL(ringbuffer.empty(), true);
  BOOST_CHECK_EQUAL(ringbuffer.full(), false);

  // Check some exceptions
  bool skip_exception_called = false;
  try { ringbuffer.skip(); }
  catch (std::out_of_range&) { skip_exception_called = true; }
  BOOST_CHECK_EQUAL(skip_exception_called, true);

  // Check arrays.  Since the ringbuffer has been emptied completely,
  // the read/write pointers should be at entry 0 again.
  BOOST_CHECK_EQUAL(ringbuffer.arrayOne().first, ringbuffer.arrayTwo().first);

  BOOST_CHECK_EQUAL(ringbuffer.arrayOne().second, 0u);
  BOOST_CHECK_EQUAL(ringbuffer.arrayTwo().second, 0u);
  BOOST_CHECK_EQUAL(ringbuffer.emptyArrayOne().second, 10u);
  BOOST_CHECK_EQUAL(ringbuffer.emptyArrayTwo().second, 0u);

  // Add some more elemets to move the read/write pointers.
  ringbuffer.write(0);
  ringbuffer.write(1);
  ringbuffer.write(2);
  ringbuffer.write(3);
  ringbuffer.write(4);
  ringbuffer.skip(2);

  // [ - | 2 3 4 - - - - - - ]
  BOOST_CHECK_EQUAL(ringbuffer.arrayOne().second, 3u);
  BOOST_CHECK_EQUAL(ringbuffer.arrayTwo().second, 0u);
  BOOST_CHECK_EQUAL(ringbuffer.emptyArrayOne().second, 6u);
  BOOST_CHECK_EQUAL(ringbuffer.emptyArrayTwo().second, 1u);

  // Add some more elemets to move the read/write pointers.
  ringbuffer.write(5);
  ringbuffer.write(6);
  ringbuffer.write(7);
  ringbuffer.write(8);
  ringbuffer.write(9);
  ringbuffer.skip(3);

  // [ - - - - | 5 6 7 8 9 - ]
  BOOST_CHECK_EQUAL(ringbuffer.arrayOne().second, 5u);
  BOOST_CHECK_EQUAL(ringbuffer.arrayTwo().second, 0u);
  BOOST_CHECK_EQUAL(ringbuffer.emptyArrayOne().second, 1u);
  BOOST_CHECK_EQUAL(ringbuffer.emptyArrayTwo().second, 4u);

  // Add some more elemets to move the read/write pointers.
  ringbuffer.write(10);
  ringbuffer.write(11);
  ringbuffer.write(12);
  ringbuffer.write(13);
  ringbuffer.skip(3);

  // [ 11 12 13 - - - - | 8 9 10 ]
  BOOST_CHECK_EQUAL(ringbuffer.arrayOne().second, 3u);
  BOOST_CHECK_EQUAL(ringbuffer.arrayTwo().second, 3u);
  BOOST_CHECK_EQUAL(ringbuffer.emptyArrayOne().second, 4u);
  BOOST_CHECK_EQUAL(ringbuffer.emptyArrayTwo().second, 0u);

  // Check that iterators behave correctly.
  size_t i = 0;
  for (typename RingBuffer<T>::iterator it = ringbuffer.begin(); it != ringbuffer.end(); ++it, ++i)
  {
    BOOST_CHECK_EQUAL(ringbuffer.at(i), *it);
  }
  // Same for postfix increment.
  i = 0;
  for (typename RingBuffer<T>::iterator it = ringbuffer.begin(); it != ringbuffer.end(); it++, ++i)
  {
    BOOST_CHECK_EQUAL(ringbuffer.at(i), *it);
  }
  // Check that const_iterators behave correctly.
  i = 0;
  for (typename RingBuffer<T>::const_iterator it = ringbuffer.begin(); it != ringbuffer.end(); ++it, ++i)
  {
    BOOST_CHECK_EQUAL(ringbuffer.at(i), *it);
  }
  // Same for postfix increment.
  i = 0;
  for (typename RingBuffer<T>::const_iterator it = ringbuffer.begin(); it != ringbuffer.end(); it++, ++i)
  {
    BOOST_CHECK_EQUAL(ringbuffer.at(i), *it);
  }

  // Check that we can move the iterator around.
  BOOST_CHECK((ringbuffer.begin() + 3) - 3 == ringbuffer.begin());
  BOOST_CHECK((ringbuffer.begin() - 3) + 3 == ringbuffer.begin());
  BOOST_CHECK_EQUAL(ringbuffer.end() - ringbuffer.begin(), 6);
  BOOST_CHECK_EQUAL(ringbuffer.begin() - ringbuffer.end(), -6);

  // Now make some more room, then add some elements externally.
  ringbuffer.skip(3);
  // [ 11 12 13 - - - - - - - | ]
  T *empty_space = ringbuffer.emptyArrayOne().first;
  for (size_t i=0; i<5; ++i)
  {
    empty_space[i] = T(i+14);
  }
  // [ 11 12 13 (14) (15) (16) (17) (18) - - | ]
  ringbuffer.fakeWrite(5);

  // [ 11 12 13 14 15 16 17 18 - - | ]
  BOOST_CHECK_EQUAL(ringbuffer.size(), 8u);
  BOOST_CHECK_EQUAL(ringbuffer.capacity(), 10u);
  BOOST_CHECK_EQUAL(ringbuffer.reserve(), 2u);
  BOOST_CHECK_EQUAL(ringbuffer.empty(), false);
  BOOST_CHECK_EQUAL(ringbuffer.full(), false);
  BOOST_CHECK_EQUAL(ringbuffer.at(0), T(11));
  BOOST_CHECK_EQUAL(ringbuffer.at(1), T(12));
  BOOST_CHECK_EQUAL(ringbuffer.at(2), T(13));
  BOOST_CHECK_EQUAL(ringbuffer.at(3), T(14));
  BOOST_CHECK_EQUAL(ringbuffer.at(4), T(15));
  BOOST_CHECK_EQUAL(ringbuffer.at(5), T(16));
  BOOST_CHECK_EQUAL(ringbuffer.at(6), T(17));
  BOOST_CHECK_EQUAL(ringbuffer.at(7), T(18));

  BOOST_CHECK_EQUAL(ringbuffer.arrayOne().second, 8u);
  BOOST_CHECK_EQUAL(ringbuffer.arrayTwo().second, 0u);
  BOOST_CHECK_EQUAL(ringbuffer.emptyArrayOne().second, 2u);
  BOOST_CHECK_EQUAL(ringbuffer.emptyArrayTwo().second, 0u);

  // Now fill it up completely.
  ringbuffer.write(T(19));
  ringbuffer.write(T(20));
  BOOST_CHECK_EQUAL(ringbuffer.size(), 10u);
  BOOST_CHECK_EQUAL(ringbuffer.capacity(), 10u);
  BOOST_CHECK_EQUAL(ringbuffer.reserve(), 0u);
  BOOST_CHECK_EQUAL(ringbuffer.empty(), false);
  BOOST_CHECK_EQUAL(ringbuffer.full(), true);
  BOOST_CHECK_EQUAL(ringbuffer.at(8), T(19));
  BOOST_CHECK_EQUAL(ringbuffer.at(9), T(20));
  BOOST_CHECK_EQUAL(ringbuffer.arrayOne().second, 10u);
  BOOST_CHECK_EQUAL(ringbuffer.arrayTwo().second, 0u);
  BOOST_CHECK_EQUAL(ringbuffer.emptyArrayOne().second, 0u);
  BOOST_CHECK_EQUAL(ringbuffer.emptyArrayTwo().second, 0u);

  // Check that we can move the iterator around.
  BOOST_CHECK((ringbuffer.begin() + 3) - 3 == ringbuffer.begin());
  BOOST_CHECK((ringbuffer.begin() - 3) + 3 == ringbuffer.begin());
  BOOST_CHECK_EQUAL(ringbuffer.end() - ringbuffer.begin(), 10);
  BOOST_CHECK_EQUAL(ringbuffer.begin() - ringbuffer.end(), -10);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(RingBufferConstness, T, TestTypes)
{
  RingBuffer<T> ringbuffer(10);

  // Add some elements until it's completely full.
  ringbuffer.write(0);
  ringbuffer.write(1);
  ringbuffer.write(2);
  ringbuffer.write(3);
  ringbuffer.write(4);
  ringbuffer.write(5);
  ringbuffer.write(6);
  ringbuffer.write(7);
  ringbuffer.write(8);
  ringbuffer.write(9);

  const RingBuffer<T>& ringbuffer_const = ringbuffer;

  BOOST_CHECK_EQUAL(ringbuffer_const.size(), 10u);
  BOOST_CHECK_EQUAL(ringbuffer_const.capacity(), 10u);
  BOOST_CHECK_EQUAL(ringbuffer_const.reserve(), 0u);
  BOOST_CHECK_EQUAL(ringbuffer_const.empty(), false);
  BOOST_CHECK_EQUAL(ringbuffer_const.full(), true);

  // Check all contained element values
  BOOST_CHECK_EQUAL(ringbuffer_const.at(0), T(0));
  BOOST_CHECK_EQUAL(ringbuffer_const.at(1), T(1));
  BOOST_CHECK_EQUAL(ringbuffer_const.at(2), T(2));
  BOOST_CHECK_EQUAL(ringbuffer_const.at(3), T(3));
  BOOST_CHECK_EQUAL(ringbuffer_const.at(4), T(4));
  BOOST_CHECK_EQUAL(ringbuffer_const.at(5), T(5));
  BOOST_CHECK_EQUAL(ringbuffer_const.at(6), T(6));
  BOOST_CHECK_EQUAL(ringbuffer_const.at(7), T(7));
  BOOST_CHECK_EQUAL(ringbuffer_const.at(8), T(8));
  BOOST_CHECK_EQUAL(ringbuffer_const.at(9), T(9));

  // Check that const_iterators behave correctly.
  size_t i = 0;
  for (typename RingBuffer<T>::const_iterator it = ringbuffer_const.begin(); it != ringbuffer_const.end(); ++it, ++i)
  {
    BOOST_CHECK_EQUAL(ringbuffer_const.at(i), *it);
  }
  // Same for postfix increment.
  i = 0;
  for (typename RingBuffer<T>::const_iterator it = ringbuffer_const.begin(); it != ringbuffer_const.end(); it++, ++i)
  {
    BOOST_CHECK_EQUAL(ringbuffer_const.at(i), *it);
  }

  // Check that we can move the iterator around.
  BOOST_CHECK((ringbuffer_const.begin() + 3) - 3 == ringbuffer_const.begin());
  BOOST_CHECK((ringbuffer_const.begin() - 3) + 3 == ringbuffer_const.begin());
  BOOST_CHECK_EQUAL(ringbuffer_const.end() - ringbuffer_const.begin(), 10);
  BOOST_CHECK_EQUAL(ringbuffer_const.begin() - ringbuffer_const.end(), -10);
}

BOOST_AUTO_TEST_SUITE_END()
