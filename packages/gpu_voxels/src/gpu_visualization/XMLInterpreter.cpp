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
#include "XMLInterpreter.h"

namespace bfs = boost::filesystem;

using boost::lexical_cast;
using boost::bad_lexical_cast;

namespace gpu_voxels {
namespace visualization {


bool XMLInterpreter::getColorFromXML(glm::vec4& color, boost::filesystem::path c_path)
{
  std::string value;
  //try to find opacity value
  float a = icl_core::config::getDefault<float>((c_path / "rgba/a").string(), 1.0f);

  if (icl_core::config::get<std::string>(c_path.string(), value))
  {
    boost::algorithm::to_lower(value);
    color = glm::vec4(colorNameToVec3(value), a);
    return true;
  }
  else
  {
    c_path /= "rgba";
    float r = 0.f, g = 0.f, b = 0.f;
    if (icl_core::config::get<float>((c_path / "r").string(), r)
        | icl_core::config::get<float>((c_path / "g").string(), g)
        | icl_core::config::get<float>((c_path / "b").string(), b))
    {
      color = glm::vec4(r, g, b, a);
      return true;
    }
    return false;
  }
}
bool XMLInterpreter::getXYZFromXML(glm::vec3& position, boost::filesystem::path c_path, float default_value =
                                        0.f)
{
  float x, y, z;
  x = y = z = default_value;
  position = glm::vec3(x, y, z);
  if (icl_core::config::get<float>((c_path / "x").string(), x)
      | icl_core::config::get<float>((c_path / "y").string(), y)
      | icl_core::config::get<float>((c_path / "z").string(), z))
  {
    position = glm::vec3(x, y, z);
    return true;
  }
  return false;
}

/**
 * Loads the specified color pair from the XML file.
 * icl_core::config::initalize(..) must be called before use.
 */
bool XMLInterpreter::getColorPairFromXML(colorPair& colors, boost::filesystem::path c_path)
{
  std::string value;
  glm::vec4 color_1, color_2;
  if (getColorFromXML(color_1, c_path / "color_1"))
  {
    colors.first = color_1;

    if (getColorFromXML(color_2, c_path / "color_2"))
    {
      colors.second = color_2;
    }
    else
    { /*If there is no second color specified use the first again*/
      colors.second = color_1;
    }
    return true;
  }
  return false;
}

bool XMLInterpreter::getDataContext(DataContext* context, std::string name)
{
  bfs::path c_path(name);

  // get the offset for the context
  bfs::path offset_path = c_path / "offset";
  glm::vec3 offset;
  getXYZFromXML(offset, offset_path);
  context->m_translation_offset = offset;
  context->m_default_prim = new Cuboid(glm::vec4(0.f, 0.f, 0.f, 1.f), glm::vec3(0), glm::vec3(1.f));

  int32_t threshold = icl_core::config::getDefault<int32_t>((c_path / "occupancy_threshold").string(), 0);

  if (threshold > MAX_PROBABILITY)
  {
    LOGGING_WARNING_C(
        Visualization,
        XMLInterpreter,
        "Occupancy_threshold of " << name << " is too big (" << threshold << "). MAX_PROBABILITY (127) is used instead." << endl);
    context->m_occupancy_threshold = MAX_PROBABILITY;
  }else if(threshold < MIN_PROBABILITY)
  {
    LOGGING_WARNING_C(
        Visualization,
        XMLInterpreter,
        "Occupancy_threshold of " << name << " is too small (" << threshold << "). MIN_PROBABILITY (-127) is used instead." << endl);
    context->m_occupancy_threshold = MIN_PROBABILITY;
  }
  else if (threshold == 0)
  {
    LOGGING_WARNING_C(Visualization, XMLInterpreter,
                      "Occupancy_threshold of " << name << " is zero." << endl);
    context->m_occupancy_threshold = 0;
  }
  else
  {
    context->m_occupancy_threshold = threshold;
  }
  bool found_something = false;
  // get the colors for all the specified types
  glm::vec4 color;
  colorPair colors;
  
  //first we look for a general color:
  std::string pt = "all_types";
    
  if (getColorFromXML(color, c_path / pt))
  {
    colors.first = color;
    colors.second = color;
    for (size_t i=0; i < MAX_DRAW_TYPES; ++i)
    {
      context->m_colors[i] = colors;
    }
    found_something = true;
  }
  else if (getColorPairFromXML(colors, c_path / pt))
  {
    for (size_t i=0; i < MAX_DRAW_TYPES; ++i)
    {
      context->m_colors[i] = colors;
    }
    found_something = true;
  }
  
  // Afterwards we allow to overwrite colors of specific types:
  for (size_t i=0; i < MAX_DRAW_TYPES; ++i)
  {
    std::string pt = "type_" + boost::lexical_cast<std::string>(i);
    
    if (getColorFromXML(color, c_path / pt))
    {
      colors.first = color;
      colors.second = color;
      context->m_colors[i] = colors;
      found_something = true;
    }
    else if (getColorPairFromXML(colors, c_path / pt))
    {
      context->m_colors[i] = colors;
      found_something = true;
    }
  }
  // if found_something == false, than no colors have been found for this voxel map
  return found_something;
}
/**
 * Fills the VoxelmapContext with colors from the XML file.
 * The colors from "voxelmap_[index]" are used.
 */
bool XMLInterpreter::getVoxelmapContext(VoxelmapContext* context, uint32_t index)
{
  std::string p = "/" + context->m_map_name;
  if (!getDataContext(context, p))
  { // if no context has been found for the name try the old naming
    p = "/voxelmap_" + boost::lexical_cast<std::string>(index);
    return getDataContext(context, p);
  }
  return true;
}

bool XMLInterpreter::getVoxellistContext(CubelistContext* context, uint32_t index)
{
  std::string p = "/" + context->m_map_name;
  if (!getDataContext(context, p))
  { // if no context has been found for the name try the old naming
    p = "/voxellist_" + boost::lexical_cast<std::string>(index);
    return getDataContext(context, p);
  }
  return true;
}

bool XMLInterpreter::getOctreeContext(CubelistContext* context, uint32_t index)
{
  std::string p = "/" + context->m_map_name;
  if (!getDataContext(context, p))
  { // if no context has been found for the name try the old naming
    p = "/octree_" + boost::lexical_cast<std::string>(index);
    return getDataContext(context, p);
  }
  return true;
}

bool XMLInterpreter::getPrimitiveArrayContext(PrimitiveArrayContext* context, uint32_t index)
{
  std::string p = "/" + context->m_map_name;
  if (!getDataContext(context, p))
  { // if no context has been found for the name try the old naming
    p = "/primitive_array_" + boost::lexical_cast<std::string>(index);
    return getDataContext(context, p);
  }
  return true;
}

bool XMLInterpreter::getVisualizerContext(VisualizerContext* con)
{
  glm::vec4 color;
  if (getColorFromXML(color, "/background"))
    con->m_background_color = color;
  if (getColorFromXML(color, "/edges"))
    con->m_edge_color = color;
  con->m_camera = getCameraFromXML();


/////////////////////////////////get all the miscellaneous stuff////////////////////////////////
  bfs::path c_path("/miscellaneous");
  con->m_min_view_dim = icl_core::config::getDefault<float>((c_path / "min_view_dim").string(), 25.f);
  c_path /= "min_xyz_value";
  con->m_min_xyz_to_draw.x = icl_core::config::getDefault<uint32_t>((c_path / "x").string(), 0);
  if (con->m_min_xyz_to_draw.x == (uint32_t)0xffffffff) { //prevents strange gaps in visualization
    con->m_min_xyz_to_draw.x = 0;
  }
  con->m_min_xyz_to_draw.y = icl_core::config::getDefault<uint32_t>((c_path / "y").string(), 0);
  if (con->m_min_xyz_to_draw.y == (uint32_t)0xffffffff) { //prevents strange gaps in visualization
    con->m_min_xyz_to_draw.y = 0;
  }
  con->m_min_xyz_to_draw.z = icl_core::config::getDefault<uint32_t>((c_path / "z").string(), 0);
  if (con->m_min_xyz_to_draw.z == (uint32_t)0xffffffff) { //prevents strange gaps in visualization
    con->m_min_xyz_to_draw.z = 0;
  }
  c_path.remove_leaf();
  c_path /= "max_xyz_value";
  uint32_t m = 0xffffffff;
  con->m_max_xyz_to_draw.x = icl_core::config::getDefault<uint32_t>((c_path / "x").string(), m);
  con->m_max_xyz_to_draw.y = icl_core::config::getDefault<uint32_t>((c_path / "y").string(), m);
  con->m_max_xyz_to_draw.z = icl_core::config::getDefault<uint32_t>((c_path / "z").string(), m);
  c_path.remove_leaf();

  con->m_interpolation_length = icl_core::config::getDefault<uint32_t>(
      (c_path / "interpolation_repeat").string(), 25.f);
  con->m_draw_edges_of_triangels = icl_core::config::getDefault<bool>(
      (c_path / "draw_edges_of_triangles").string(), false);
  con->m_draw_filled_triangles = icl_core::config::getDefault<bool>((c_path / "draw_filled_triangles").string(),
                                                                  true);
  con->m_draw_whole_map = icl_core::config::getDefault<bool>((c_path / "draw_whole_map").string(), true);

  con->m_grid_distance = icl_core::config::getDefault<float>((c_path / "grid_distance").string(), 10.f);
  con->m_grid_height = icl_core::config::getDefault<float>((c_path / "grid_height").string(), 0.f);
  con->m_grid_max_x = icl_core::config::getDefault<uint32_t>((c_path / "grid_max_x").string(), 1000);
  con->m_grid_max_y = icl_core::config::getDefault<uint32_t>((c_path / "grid_max_y").string(), 1000);

  con->m_scale_unit = getUnitScale();

  if (getColorFromXML(color, (c_path / "grid_color").string()))
  {
    con->m_grid_color = color;
  }

  return true;
}

size_t XMLInterpreter::getMaxMem()
{
  bfs::path c_path("/miscellaneous");
  return icl_core::config::getDefault<uint32_t>((c_path / "max_memory_usage").string(), 0) * 1e+006; // in MByte
}
float XMLInterpreter::getMaxFps()
{
  bfs::path c_path("/miscellaneous");
  return icl_core::config::getDefault<uint32_t>((c_path / "max_fps").string(), 0);
}

std::pair<float, std::string> XMLInterpreter::getUnitScale()
{
  bfs::path c_path("/miscellaneous");
  std::pair<float, std::string> res(1, "cm");

  std::string unit_scale;

  if (!icl_core::config::get<std::string>((c_path / "unit_scale").string(), unit_scale))
  {
    LOGGING_WARNING_C(Visualization, XMLInterpreter, "No unit scale defined. Using 1 cm instead."<< endl);
    return res;
  }

  std::vector<std::string> tokens;
  boost::split(tokens, unit_scale, boost::is_any_of(" "));
  if (tokens.size() >= 2)
  {
    float c = 1.f;
    try
    {
      c = lexical_cast<float>(tokens[0]);

    } catch (bad_lexical_cast &)
    {
      LOGGING_ERROR_C(
          Visualization, XMLInterpreter,
          "Couldn't read the arguments of unit_scale correctly! Using 1 cm instead. " << tokens[0] << endl);
      return res;
    }
    res.first = c;
    res.second = tokens[1];
  }
  else
  {
    LOGGING_ERROR_C(Visualization, XMLInterpreter,
                    "Too few arguments in unit scale. Should look like \"SCALE UNIT\"." << endl);
  }
  return res;
}

bool XMLInterpreter::getSphereFromXML(Sphere*& sphere, std::string name)
{
  boost::filesystem::path c_path(name);

  glm::vec3 position;
  glm::vec4 color = glm::vec4(1.f, 0.f, 0.f, 1.f);
  if (getXYZFromXML(position, c_path))
  {
    if (!getColorFromXML(color, c_path))
    {
      LOGGING_WARNING_C(Visualization, XMLInterpreter,
                        "No color specified for sphere. Red is used instead." << endl);
    }
    float radius = icl_core::config::getDefault<float>((c_path / "radius").string(), 1.f);
    uint32_t resolution = icl_core::config::getDefault<uint32_t>((c_path / "resolution").string(), 16.f);
    sphere = new Sphere(color, position, radius, resolution);
    return true;
  }
  return false;
}
bool XMLInterpreter::getCuboidFromXML(Cuboid*& cuboid, std::string name)
{
  boost::filesystem::path c_path(name);

  glm::vec3 position;
  glm::vec3 side_length;
  glm::vec4 color = glm::vec4(1.f, 0.f, 0.f, 1.f);
  if (getXYZFromXML(position, c_path / "position"))
  {
    if (!getColorFromXML(color, c_path))
    {
      LOGGING_WARNING_C(Visualization, XMLInterpreter,
                        "No color specified for cube. Red is used instead." << endl);
    }
    if (!getXYZFromXML(side_length, c_path / "side_length", 1.f))
    {
      LOGGING_WARNING_C(Visualization, XMLInterpreter,
                        "No side length specified for cube. (1,1,1) used instead" << endl);
    }

    cuboid = new Cuboid(color, position, side_length);
    return true;
  }
  return false;
}

void XMLInterpreter::getPrimtives(std::vector<Primitive*>& primitives)
{
  Sphere* sphere;
  if (getSphereFromXML(sphere, "/sphere"))
  {
    primitives.push_back(sphere);
  }
  Cuboid* cuboid;
  if (getCuboidFromXML(cuboid, "/cuboid"))
  {
    primitives.push_back(cuboid);
  }
}
void XMLInterpreter::getDefaultSphere(Sphere*& sphere)
{
  if (!getSphereFromXML(sphere, "/defaultSphere"))
  {
    sphere = new Sphere(glm::vec4(1, 0, 0, 1), glm::vec3(0, 0, 0), 1.f, 16);
  }
}
void XMLInterpreter::getDefaultCuboid(Cuboid*& cuboid)
{
  if (!getCuboidFromXML(cuboid, "/defaultCuboid"))
  {
    cuboid = new Cuboid(glm::vec4(1, 0, 0, 1), glm::vec3(0, 0, 0), glm::vec3(1.f));
  }
}
/**
 * Loads the camera parameter from the XML file
 * icl_core::config::initalize(..) must be called before use.
 * icl_core::logging::initialize(..) implicitly calls icl_core::config::initialize
 */
Camera_gpu * XMLInterpreter::getCameraFromXML()
{
  boost::filesystem::path c_path("/camera");
  glm::vec3 camera_position = glm::vec3(
      icl_core::config::getDefault<float>((c_path / "position/x").string(), -100.f),
      icl_core::config::getDefault<float>((c_path / "position/y").string(), -100.f),
      icl_core::config::getDefault<float>((c_path / "position/z").string(), 100.f));

  glm::vec3 camera_focus; //= glm::vec3(icl_core::config::getDefault<float>((c_path / "focus/x").string(), 0.f),
//                                     icl_core::config::getDefault<float>((c_path / "focus/y").string(), 0.f),
//                                     icl_core::config::getDefault<float>((c_path / "focus/z").string(), 0.f));

  getXYZFromXML(camera_focus, c_path / "focus", -0.000001f);
  if (camera_position == camera_focus)
  {
    LOGGING_ERROR_C(
        Visualization, XMLInterpreter,
        "Camera position and focus point are equal! Reducing focus point by 10 in each dimension." << endl);
    camera_focus = camera_focus - glm::vec3(10);
  }

  float hor_angle = glm::radians(icl_core::config::getDefault<float>((c_path / "horizontal_angle").string(), 135));
  float vert_angle = glm::radians(icl_core::config::getDefault<float>((c_path / "vertical_angle").string(), -10));
  float fov = glm::radians(icl_core::config::getDefault<float>((c_path / "field_of_view").string(), 60));

  Camera_gpu::CameraContext context = Camera_gpu::CameraContext(
      camera_position, camera_focus, hor_angle, vert_angle, fov);

  float width = icl_core::config::getDefault<float>((c_path / "window_width").string(), 1024.f);
  float height = icl_core::config::getDefault<float>((c_path / "window_height").string(), 768.f);

  return new Camera_gpu(width, height, context);

}
/**
 * Converts a color into his rgb representation
 */
glm::vec3 XMLInterpreter::colorNameToVec3(std::string name)
{
  if (name.compare("black") == 0)
  {
    return glm::vec3(0.f);
  }
  else if (name.compare("white") == 0)
  {
    return glm::vec3(1.f, 1.f, 1.f);
  }
  else if (name.compare("red") == 0)
  {
    return glm::vec3(1.f, 0.f, 0.f);
  }
  else if (name.compare("dark red") == 0)
  {
    return glm::vec3(0.5f, 0.f, 0.f);
  }
  else if (name.compare("green") == 0)
  {
    return glm::vec3(0.f, 1.f, 0.f);
  }
  else if (name.compare("dark green") == 0)
  {
    return glm::vec3(0.f, .5f, 0.f);
  }
  else if (name.compare("blue") == 0)
  {
    return glm::vec3(0.f, 0.f, 1.f);
  }
  else if (name.compare("dark blue") == 0)
  {
    return glm::vec3(0.f, 0.f, 0.5f);
  }
  else if (name.compare("gray") == 0)
  {
    return glm::vec3(.75f, .75f, .75f);
  }
  else if (name.compare("dark gray") == 0)
  {
    return glm::vec3(.5f, .5f, .5f);
  }
  else if (name.compare("yellow") == 0)
  {
    return glm::vec3(1.f, 1.f, 0.f);
  }
  else if (name.compare("dark yellow") == 0)
  {
    return glm::vec3(0.5f, 0.5f, 0.f);
  }
  else if (name.compare("cyan") == 0)
  {
    return glm::vec3(0.f, 1.f, 1.f);
  }
  else if (name.compare("dark cyan") == 0)
  {
    return glm::vec3(0.f, .5f, .5f);
  }
  else if (name.compare("magenta") == 0)
  {
    return glm::vec3(1.f, 0.f, 1.f);
  }
  else if (name.compare("dark magenta") == 0)
  {
    return glm::vec3(0.5f, 0.f, 0.5f);
  }
  LOGGING_WARNING_C(Visualization, XMLInterpreter,
                    "Specified color \"" << name << "\" is not supported. Black is used instead." << endl);
  return glm::vec3(0.f);
}

} // end of ns
} // end of ns
