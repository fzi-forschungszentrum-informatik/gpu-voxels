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
 * \date    2007-06-10
 *
 */
//----------------------------------------------------------------------

#include <cppunit/TextOutputter.h>
#include <cppunit/CompilerOutputter.h>
#include <cppunit/XmlOutputter.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/TestRunner.h>
#include <cppunit/extensions/TestFactoryRegistry.h>

#include <icl_core/os_lxrt.h>
#include <icl_core_logging/Logging.h>

#include "icl_core_testsuite/TestListener.h"
#include "icl_core_testsuite/test_suite.h"

namespace icl_core {

int runCppunitTestSuite(TestResultOutputType outputType)
{
  icl_core::os::lxrtStartup();
  icl_core::logging::initialize();

  // Informiert Test-Listener ueber Testresultate
  CPPUNIT_NS::TestResult testresult;

  // Listener zum Sammeln der Testergebnisse registrieren
  CPPUNIT_NS::TestResultCollector collectedresults;
  testresult.addListener(&collectedresults);

  // Test-Suite ueber die Registry im Test-Runner einfuegen
  CPPUNIT_NS::TestRunner testrunner;
  testrunner.addTest(CPPUNIT_NS::TestFactoryRegistry::getRegistry().makeTest());

  // Resultate ausgeben
  switch (outputType)
  {
    case eTROT_Text:
    {
      icl_core::TestListener progress;
      testresult.addListener(&progress);

      testrunner.run(testresult);

      CPPUNIT_NS::TextOutputter textoutputter(&collectedresults, std::cerr);
      textoutputter.write();
      break;
    }
    case eTROT_Compiler:
    {
      icl_core::TestListener progress;
      testresult.addListener(&progress);

      testrunner.run(testresult);

      CPPUNIT_NS::CompilerOutputter compileroutputter(&collectedresults, std::cerr);
      compileroutputter.write();
      break;
    }
    case eTROT_Xml:
    {
      testrunner.run(testresult);

      CPPUNIT_NS::XmlOutputter xmloutputter(&collectedresults, std::cerr, "UTF-8");
      xmloutputter.write();
      break;
    }
  }

  icl_core::logging::shutdown();
  icl_core::os::lxrtShutdown();

  // Rueckmeldung, ob Tests erfolgreich waren
  return collectedresults.wasSuccessful() ? 0 : 1;
}

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

int RunCppunitTestSuite(TestResultOutputType outputType)
{
  return runCppunitTestSuite(outputType);
}

#endif
/////////////////////////////////////////////////

}
