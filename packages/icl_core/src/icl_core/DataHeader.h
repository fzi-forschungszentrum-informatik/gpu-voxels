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
 * \author  Sebastian Klemm <klemm@fzi.de>
 * \date    2014-04-01
 *
 *  The structs DataHeader and Stamped contained in this file may be
 *  used to provide frequently required meta data that may be attached
 *  to any kind of data.
 *
 *  Included are: a coordinate system, a time stamp and a sequence
 *  number.
 *
 *  A way to attach the DataHeader to a class MyClass would be:
 *
 *  typedef Stamped<MyClass> MyClassStamped;
 *
 *  Shared pointers are already included within the Stamped struct.
 *  For convenience the data field of the Stamped struct may be
 *  accessed via operator * () or operator -> ().
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_DATA_HEADER_H_INCLUDED
#define ICL_CORE_DATA_HEADER_H_INCLUDED

#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/date_time.hpp>
#include <boost/version.hpp>
#if (BOOST_VERSION >= 104800)
# include <boost/type_traits/has_left_shift.hpp>
#endif
#include <icl_core/ImportExport.h>
#include <icl_core/BaseTypes.h>

#ifdef _IC_BUILDER_EIGEN_
#include <Eigen/Core>
#endif

namespace icl_core {

/*! A generic header which can be used to associate a coordinate
 *  system ID, a timestamp, a sequence number and a device-specific
 *  item number with any kind of datum.
 *  \see icl_core::Stamped
 */
struct DataHeader
{
  /*! Default constructor.  Sets an empty coordinate system
   *  identifier, an invalid timestamp and a sequence number of zero.
   */
  DataHeader()
    : coordinate_system(),
      timestamp(),
      sequence_number(0)
  { }

  //! Constructor.
  DataHeader(const std::string& coordinate_system,
             const boost::posix_time::ptime& timestamp,
             uint32_t sequence_number = 0)
    : coordinate_system(coordinate_system),
      timestamp(timestamp),
      sequence_number(sequence_number)
  { }

  //! Destructor.
  ~DataHeader()
  { }

  //! An identifier for the coordinate system used.
  std::string coordinate_system;
  //! A timestamp, e.g. when data was recorded / has to be considered.
  boost::posix_time::ptime timestamp;
  //! A sequence number to distinguish data with identical timestamps.
  uint32_t sequence_number;
  /*! A device-specific item number.  This is useful if the data items
   *  already come with some kind of internal numbering, e.g. the
   *  internal frame count from a camera sensor.
   *  \see icl_core::sourcesink::DSIN
   */
  std::size_t dsin;
};


/*! A base struct for the Stamped struct without dependency on the
 *  template parameter.
 */
struct StampedBase
{
  //! Convenience shared pointer typedefs
  typedef boost::shared_ptr<StampedBase> Ptr;
  typedef boost::shared_ptr<const StampedBase> ConstPtr;

  //! define default Constructor and Destructor for boost_dynamic_cast
  StampedBase()
  {}

  virtual ~StampedBase()
  {}

  /*! virtual print function that is used in the streaming operator and will be
   *  overloaded by Stamped<T>
   */
  virtual void print(std::ostream& os) const = 0;

  //! Access the data header.
  virtual DataHeader& header() = 0;
  //! Access the data header (const version).
  virtual const DataHeader& header() const = 0;
};

namespace internal {

/*! Helper class which produces output for objects of type \a T only
 *  if it is known at compile time that a suitable stream operator is
 *  available.  This is used by icl_core::Stamped<DataType>.  Note
 *  that there are some (rare) cases where this might not work as
 *  expected (see the Boost TypeTraits documentation for details).
 *  \see icl_core::Stamped<DataType>::print(std::ostream&) const
 */
template <typename T, bool has_left_shift>
struct ToStream;

template <typename T>
struct ToStream<T, false>
{
  static std::ostream& print(std::ostream& os, const T& obj)
  {
    return os << sizeof(T) << " data bytes";
  }
};

template <typename T>
struct ToStream<T, true>
{
  static std::ostream& print(std::ostream& os, const T& obj)
  {
    return os << obj;
  }
};

}

/*! A generic wrapper for combining any kind of data with a
 *  DataHeader.
 */
template <class DataType>
struct Stamped : public StampedBase
{
private:
  // IMPORTANT: Do not change the order (data -> header) as it is
  // important for memory alignment!

  //! The wrapped data.
  DataType m_data;
  //! The data header.
  DataHeader m_header;

public:
  /* If we build against Eigen it is possible that the DataType depends on
   * Eigen. Then the Stamped should be aligned.
   */
#ifdef _IC_BUILDER_EIGEN_
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#endif

  //! Convenience smart pointer typedef.
  typedef boost::shared_ptr<Stamped<DataType> > Ptr;
  //! Convenience smart pointer typedef.
  typedef boost::shared_ptr<const Stamped<DataType> > ConstPtr;

  //! Default constructor which leaves the data uninitialized.
  Stamped()
    : m_header()
  { }

  //! Implicit construction from a data value.
  Stamped(const DataType& data)
    : m_data(data),
      m_header()
  { }

  //! Implicit construction from a data header.
  Stamped(const DataHeader& header)
    : m_header(header)
  { }

  //! Full constructor.
  Stamped(const DataType& data, const DataHeader& header)
    : m_data(data),
      m_header(header)
  { }

  //! Implicit conversion back to the original data type.
  inline operator DataType () const { return m_data; }

  //! Access the data header.
  virtual DataHeader& header() { return m_header; }
  //! Access the data header (const version).
  virtual const DataHeader& header() const  { return m_header; }

  /*! Access to the wrapped data.  While direct access to the #data
   *  member is allowed, the access functions should be preferred.
   */
  inline DataType& get() { return m_data; }
  /*! Access to the wrapped data.  While direct access to the #data
   *  member is allowed, the access functions should be preferred.
   */
  inline const DataType& get() const { return m_data; }
  /*! Explicitly const access to the wrapped data.  This is in
   *  anticipation of C++11, to allow correct type deduction for the
   *  auto keyword.
   */
  inline const DataType& cget() const { return m_data; }

  //! Quick access to the wrapped data.
  inline DataType& operator * () { return m_data; }
  //! Quick access to the wrapped data.
  inline const DataType& operator * () const { return m_data; }

  //! Quick access to the wrapped data.
  inline DataType *operator -> () { return &m_data; }
  //! Quick access to the wrapped data.
  inline const DataType *operator -> () const { return &m_data; }

  /*! Prints the contents to the given output stream.  Note that this
   *  method tries to produce output for #data only if it is known at
   *  compile time that a suitable stream operator is available.  This
   *  relies on the boost::has_left_shift<> type trait.  Note that
   *  there are some (rare) cases where this might not work as
   *  expected (see the Boost TypeTraits documentation for details):
   *
   *  - The class cannot detect if the operator<< is private.  If it
   *    is, then you will get a compiler error.
   *  - There is an issue if the operator exists only for type A and B
   *    is convertible to A. In this case, the compiler will report an
   *    ambiguous overload.
   *  - There is an issue when applying this trait to template
   *    classes. If operator<< is defined but does not bind for a
   *    given template type, it is still detected by the trait which
   *    returns true instead of false, resulting in a compiler error.
   *  - The volatile qualifier is not properly handled and would lead
   *    to undefined behavior.
   *
   *  \see StampedBase#print()
   */
  virtual void print(std::ostream& os) const
  {
    os << "[seq: " << this->m_header.sequence_number
       << ", timestamp: " << boost::posix_time::to_simple_string(this->m_header.timestamp)
       << ", DSIN: " << this->m_header.dsin
       << ", Frame: " << m_header.coordinate_system
       << "; ";
#if (BOOST_VERSION >= 104800)
    internal::ToStream<DataType, boost::has_left_shift<std::ostream, DataType>::value>::print(os, m_data);
#else
    // Boost versions prior to 1.48 do not have the has_left_shift
    // type trait.
    internal::ToStream<DataType, false>::print(os, m_data);
#endif
    os << "]";
  }
};

/*! operator << to stream StampedBase to an ostream. Internally it calls the
 *  virtual print() function that is overloaded by the Stamped<T>
 *  implementation.
 */
ICL_CORE_IMPORT_EXPORT std::ostream& operator << (std::ostream& os, StampedBase const& stamped);

}

#endif
