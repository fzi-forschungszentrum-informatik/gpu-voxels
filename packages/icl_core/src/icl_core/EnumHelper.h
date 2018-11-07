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
 * \date    2009-03-08
 *
 * \brief   Contains helper functions to handle enums with textual descriptions.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_ENUM_HELPER_H_INCLUDED
#define ICL_CORE_ENUM_HELPER_H_INCLUDED

#include <vector>

#include "icl_core/BaseTypes.h"
#include "icl_core/ImportExport.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {

/*! Converts a string to an enumeration value. The descriptions for
 *  the individual enumeration values have to be provided in an array,
 *  which is terminated by a \c NULL entry.
 *
 *  \param str A string, which should be converted into an enumeration
 *         value.
 *  \param value The enumeration value into which the result will be
 *         written.
 *  \param descriptions An array of descriptions for the enumeration
 *         values, terminated by the \a end_marker.
 *  \param end_marker The end marker used to terminate the description
 *         array.
 *
 *  \returns \c true if \a str has been found in \a descriptions. In
 *           this case \a value is set to the corresponding index. \c
 *           false if \a str could not be found in \a descriptions.
 */
bool ICL_CORE_IMPORT_EXPORT string2Enum(const String& str, int32_t& value,
                                        const char * const *descriptions,
                                        const char *end_marker = NULL);
bool ICL_CORE_IMPORT_EXPORT string2Enum(const String& str, int32_t& value,
                                        const std::vector<std::string>& descriptions);
bool ICL_CORE_IMPORT_EXPORT string2Enum(const String& str, int64_t& value,
                                        const std::vector<std::string>& descriptions);

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

/*! Converts a string to an enumeration value. The descriptions for
 *  the individual enumeration values have to be provided in an array,
 *  which is terminated by a \c NULL entry.
 *  \deprecated Obsolete coding style.
 */
bool ICL_CORE_IMPORT_EXPORT ICL_CORE_VC_DEPRECATE_STYLE
String2Enum(const String& str, int32_t& value, const char * const *descriptions,
            const char *end_marker = NULL) ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
/////////////////////////////////////////////////

}

#endif
