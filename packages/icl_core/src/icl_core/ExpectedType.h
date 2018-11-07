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
 * \date    2014-04-05
 *
 * Helper classes to allow reasoning about the expected type of a
 * function returned at runtime, without actually creating an object
 * of that type.
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_EXPECTED_TYPE_H_INCLUDED
#define ICL_CORE_EXPECTED_TYPE_H_INCLUDED

#include <boost/shared_ptr.hpp>

namespace icl_core {

template <typename T> struct ExpectedTypeIs;

/*! An object bearing information about the type expected from some
 *  operation.  If a complex function call will return an object of a
 *  certain type depending on runtime circumstances, it may not always
 *  be desirable, or even possible, to start the operation, wait for
 *  the returned object, and perform a dynamic_cast to see what it is.
 *  Instead, the class can provide another function which tells you
 *  which datatype to expect given the parameters you would pass to
 *  the complex function.  This can be used as follows:
 *
 *  \code
 *  class Foo : public BaseClass { ... };
 *  class Bar : public BaseClass { ... };
 *  ExpectedType::Ptr et = complex_op.expectedReturnType(params);
 *  if (et->is<Foo>()) std::cout << "Operation will return a Foo object\n";
 *  if (et->is<Bar>()) std::cout << "Operation will return a Bar object\n";
 *  BaseClass::Ptr result = complex_op.execute(params);
 *  // Wait some time...
 *  std::cout << result << "\n"; // Prints stuff for Foo or Bar, depending on result
 *  \endcode
 *
 *  \see ExpectedTypeIs<T> for how to write an expectedReturnType
 *       function as used in the code example.
 */
struct ExpectedType
{
public:
  typedef boost::shared_ptr<const ExpectedType> Ptr;

  virtual ~ExpectedType() { }

  //! Check if an actual type equals this expected type.
  template <typename T> bool is() const { return ExpectedTypeIs<T>().equals(this); }

  //! Directly compare two ExpectedType objects.
  bool operator == (const ExpectedType& other) const
  {
    return this->equals(&other);
  }

  //! Directly compare two ExpectedType objects.
  bool operator != (const ExpectedType& other) const
  {
    return !this->equals(&other);
  }

protected:
  virtual bool equals(const ExpectedType *other) const = 0;
};

/*! An object bearing the information that the expected type of some
 *  operation is \a T.  If you have a complex function which will
 *  return an object of a certain type depending on runtime
 *  circumstances, you can write another function that will quickly
 *  inspect the parameters to provide information on the type returned
 *  by the complex function.  Usage example:
 *  \code
 *  class Foo : public BaseClass { ... };
 *  class Bar : public BaseClass { ... };
 *  class ComplexOp
 *  {
 *    ...
 *    // The complex operation takes a long time and returns either a
 *    // Foo or a Bar depending on the parameter
 *    BaseClass::Ptr complexOperation(int parameter);
 *
 *    // Informs the user quickly what type will be provided.
 *    ExpectedType::Ptr complexOperationResult(int parameter)
 *    {
 *      if (parameter > 0) return ExpectedType::Ptr(new ExpectedTypeIs<Foo>);
 *      else return ExpectedType::Ptr(new ExpectedTypeIs<Bar>);
 *    }
 *  };
 *  \endcode
 *
 *  \see ExpectedType for how to use the expected type result.
 */
template <typename T>
struct ExpectedTypeIs : public ExpectedType
{
protected:
  bool equals(const ExpectedType *other) const
  {
    return dynamic_cast<const ExpectedTypeIs<T> *>(other) != NULL;
  }
  friend struct ExpectedType;
};

}

#endif
