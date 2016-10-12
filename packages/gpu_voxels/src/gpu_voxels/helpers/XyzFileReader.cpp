// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
//
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE.txt in the top
// directory of the source code.
//
// © Copyright 2014 FZI Forschungszentrum Informatik, Karlsruhe, Germany
//
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Andreas Hermann
 * \date    2014-06-12
 *
 */
//----------------------------------------------------------------------
#include <gpu_voxels/helpers/XyzFileReader.h>
#include <gpu_voxels/helpers/common_defines.h>

namespace gpu_voxels {
namespace file_handling {

bool XyzFileReader::readPointCloud(const std::string filename, std::vector<Vector3f> &points)
{
  float x;
  float y;
  float z;
  std::string line;
  std::ifstream file(filename.c_str());

  if (!file)
  {
    LOGGING_ERROR(Gpu_voxels_helpers, "Could not open file " << filename.c_str() << " !" << endl);
    return false;
  }
  while (std::getline(file, line))
  {
    std::istringstream iss(line);
    Vector3f vec;
    // reads the float value from the file and adds the offset.
    while (iss >> x && iss >> y && iss >> z)
    {
      vec.x = x;
      vec.y = y;
      vec.z = z;
      points.push_back(vec);
    }
  }
  LOGGING_DEBUG(
      Gpu_voxels_helpers,
      "XYZ-FileReader: loaded " << points.size() << " points ("<< (points.size()*sizeof(Vector3f)) * cBYTE2MBYTE << " MB on CPU) from "<< filename.c_str() << "." << endl);
  file.close();
  return true;
}


} // end of namespace
} // end of namespace
