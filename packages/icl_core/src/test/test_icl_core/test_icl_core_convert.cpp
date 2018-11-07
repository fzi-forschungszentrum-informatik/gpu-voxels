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
 * \date    2015-01-30
 *
 * Test/example program for icl_core::convert().
 */
//----------------------------------------------------------------------
#include <icl_core/Convert.h>

#include <iostream>

struct Foo
{
  Foo() : foo(42) { }
  int foo;
};

struct Bar
{
  Bar() : bar(23) { }
  int bar;
};

std::ostream& operator << (std::ostream& os, const Foo& foo)
{
  return os << "Foo:" << foo.foo;
}

std::ostream& operator << (std::ostream& os, const Bar& bar)
{
  return os << "Bar:" << bar.bar;
}

namespace icl_core {

template <>
void convert<>(const Foo& from, Bar& to)
{
  to.bar = from.foo;
}

template <>
void convert<>(const Bar& from, Foo& to)
{
  to.foo = from.bar;
}

}

int main()
{
  Foo foo;
  Bar bar;

  Bar bar2 = icl_core::convert<Bar>(foo);
  Foo foo2 = icl_core::convert<Foo>(bar);

  std::cout << "Original objects:  " << foo << " " << bar << "\n"
            << "Converted objects: " << foo2 << " " << bar2 << std::endl;
  return 0;
}
