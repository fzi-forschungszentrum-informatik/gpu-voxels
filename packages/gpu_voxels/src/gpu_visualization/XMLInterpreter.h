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
 * \author  Matthias Wagner
 * \date    2014-01-20
 *
 *  \brief The interpreter for the xml config file.
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_VISUALIZATION_XMLINTERPRETER_H_INCLUDED
#define GPU_VOXELS_VISUALIZATION_XMLINTERPRETER_H_INCLUDED

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

#include <icl_core_config/Config.h>

#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_visualization/Camera.h>

#include <gpu_visualization/VisualizerContext.h>
#include <gpu_visualization/Cuboid.h>
#include <gpu_visualization/Sphere.h>

#include <gpu_visualization/logging/logging_visualization.h>


namespace gpu_voxels {
namespace visualization {

class XMLInterpreter
{
public:

  XMLInterpreter()
  {
    m_is_initialized = false;
  }

  bool initialize(int& argc, char *argv[])
  {
    m_is_initialized = icl_core::config::initialize(argc, argv);
    return m_is_initialized;
  }

  bool getVisualizerContext(VisualizerContext* con);
  bool getVoxelmapContext(VoxelmapContext* context, uint32_t index = 0);
  bool getVoxellistContext(CubelistContext *context, uint32_t index = 0);
  bool getOctreeContext(CubelistContext *context, uint32_t index);
  bool getPrimitiveArrayContext(PrimitiveArrayContext* context, uint32_t index);

  void getPrimtives(std::vector<Primitive*>& primitives);
  void getDefaultSphere(Sphere*& sphere);
  void getDefaultCuboid(Cuboid*& cuboid);

  size_t getMaxMem();
  float getMaxFps();

private:
  /////////////////////////////private functions//////////////////////////////////
  bool getXYZFromXML(glm::vec3& position, boost::filesystem::path c_path, float default_value);

  glm::vec3 colorNameToVec3(std::string name);
  bool getColorFromXML(glm::vec4& color, boost::filesystem::path c_path);
  bool getColorPairFromXML(colorPair& colors, boost::filesystem::path c_path);
  std::pair<float, std::string> getUnitScale();

  bool getDataContext(DataContext* context, std::string name);

  bool getSphereFromXML(Sphere*& sphere, std::string name);
  bool getCuboidFromXML(Cuboid*& cuboid, std::string name);
  Camera_gpu* getCameraFromXML();

  /////////////////////////////private variables/////////////////////////////////////initialized
  bool m_is_initialized;
};

} // end of ns
} // end of ns

#endif
