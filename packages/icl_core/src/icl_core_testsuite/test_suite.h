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
 * \date    2008-01-06
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_TESTSUITE_TEST_SUITE_H_INCLUDED
#define ICL_CORE_TESTSUITE_TEST_SUITE_H_INCLUDED

#include "icl_core_testsuite/ImportExport.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {

enum TestResultOutputType
{
  eTROT_Text,     /*!< Output on stderr with a textual overview and
                   *   summary. Optimized for human readability.
                   */
  eTROT_Compiler, /*!< Output on stderr with a textual overview and
                   * summary. Compiler-like output of assertions to
                   * help your IDE parse the output.
                   */
  eTROT_Xml       /*!< Output on stderr in xml format */
};

ICL_CORE_TESTSUITE_IMPORT_EXPORT int runCppunitTestSuite(TestResultOutputType outputType = eTROT_Compiler);

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

int ICL_CORE_TESTSUITE_IMPORT_EXPORT ICL_CORE_VC_DEPRECATE_STYLE RunCppunitTestSuite(TestResultOutputType outputType = eTROT_Compiler)
  ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
/////////////////////////////////////////////////

}

#endif
