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
 * \date    2009-09-08
 *
 * \brief   Contains global filesystem functions
 *
 */
//----------------------------------------------------------------------
#include "icl_core/os_fs.h"

#include <iostream>

#include "icl_core/BaseTypes.h"

#ifdef _IC_BUILDER_ZLIB_
# include <zlib.h>
#endif

namespace icl_core {
namespace os {

#ifdef _IC_BUILDER_ZLIB_
bool zipFile(const char *filename, const char *additional_extension)
{
  bool ret = true;
  icl_core::String gzip_file_name = icl_core::String(filename) + additional_extension + ".gz";
  char big_buffer[0x1000];
  int bytes_read = 0;
  gzFile unzipped_file = gzopen(filename, "rb");
  gzFile zipped_file = gzopen(gzip_file_name.c_str(), "wb");

  if (unzipped_file != NULL && zipped_file != NULL)
  {
    bytes_read = gzread(unzipped_file, big_buffer, 0x1000);
    while (bytes_read > 0)
    {
      if (gzwrite(zipped_file, big_buffer, bytes_read) != bytes_read)
      {
        std::cerr << "ZipFile(" << filename << "->" << gzip_file_name << ") Error on writing." << std::endl;
        ret = false;
        break;
      }

      bytes_read = gzread(unzipped_file, big_buffer, 0x1000);
    }
  }

  if (unzipped_file != NULL)
  {
    gzclose(unzipped_file);
  }
  if (zipped_file != NULL)
  {
    gzclose(zipped_file);
  }

  return ret;
}

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

bool ZipFile(const char *filename, const char *additional_extension)
{
  return zipFile(filename, additional_extension);
}

#endif
/////////////////////////////////////////////////

#endif

}
}
