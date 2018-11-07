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
 * \date    2007-12-07
 *
 * \brief   Contains ConfigIterator.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_CONFIG_CONFIG_ITERATOR_H_INCLUDED
#define ICL_CORE_CONFIG_CONFIG_ITERATOR_H_INCLUDED

#include "icl_core/KeyValueDirectory.h"
#include "icl_core_config/ImportExport.h"

#ifdef _SYSTEM_WIN32_
#include "icl_core/KeyValueDirectory.hpp"
#endif

namespace icl_core {

#ifdef _SYSTEM_WIN32_
#ifdef __INSURE__
// ParaSoft Insure++ produces linker errors when the class itself is
// instantiated with the ICL_CORE_CONFIG_IMPORT_EXPORT modifier.  We
// therefore add declarations for all exported class members.
template ICL_CORE_CONFIG_IMPORT_EXPORT
KeyValueDirectoryIterator<String>::KeyValueDirectoryIterator(const String& query,
                                                             const KeyValueDirectory<String> *directory);
template ICL_CORE_CONFIG_IMPORT_EXPORT
String KeyValueDirectoryIterator<String>::key() const;
template ICL_CORE_CONFIG_IMPORT_EXPORT
String KeyValueDirectoryIterator<String>::matchGroup(size_t index) const;
template ICL_CORE_CONFIG_IMPORT_EXPORT
bool KeyValueDirectoryIterator<String>::next();
template ICL_CORE_CONFIG_IMPORT_EXPORT
void KeyValueDirectoryIterator<String>::reset();
template ICL_CORE_CONFIG_IMPORT_EXPORT
ConvertToRef<String>::ToConstRef KeyValueDirectoryIterator<String>::value() const;
#else
template ICL_CORE_CONFIG_IMPORT_EXPORT class KeyValueDirectoryIterator<String>;
#endif
#endif

namespace config {

typedef icl_core::KeyValueDirectoryIterator<icl_core::String> ConfigIterator;

}
}

#endif
