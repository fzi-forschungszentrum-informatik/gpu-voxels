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

/*!
 * \brief The XMLInterpreter class is reading XML config files given to the gpu_voxels_visualizer by the -c argument.
 * Here is an overview, on what can be configured. Please see description of this classes funtions for the XML syntax.
 * - Backgound color
 * - Mesh edges color
 * - Initial camera pose
 *   - Orbit
 *   - Free flight mode
 *   - Field of view
 * - Grid properties
 *   - Color
 *   - Size
 *   - Resolution
 * - Color of maps
 *   - Single color
 *   - Color gradient (with interpolation distance)
 *   - Color per Voxel meaning
 *   - Color per occupancy value
 * - Occupancy thresholds per map
 * - Cutoff dimensions to create slices through a scene
 * - Primitive list colors
 * - Maximum allowed memory usage
 * - Maximum FPS
 * - Drawing style (edges, filled polygons)
 * - Conversion factor from Voxels to metric size
 *   (used when displaying Voxel info on console)
 * - Window size
 * - View truncation
 *   - Toggle view truncation
 *   - Define min size of viewing cube
 */
class XMLInterpreter
{
public:

  XMLInterpreter()
  {
  }

  /*!
   * \brief getVisualizerContext is the main function to read parameters.
   * It is called from the other <code>get...Context()</code> functions.
   * Please see description of this classes private funtions for the XML syntax of the child elements.
   * All paramters have to be included in a <code><visualizer_context></code>-Environment
   * The context specific atomic parameters are listed here:
   * - <code><background> color </background></code> defines background color
   * - <code><edges> color </edges></code> defines the color of the Voxel edges.
   * - <code><camera> camera parameters </camera></code> Threshold at which voxels get drawn
   * \param con The context to configure
   * \return true if context was found
   */
  bool getVisualizerContext(VisualizerContext* con);

  /*!
   * \brief getVoxelmapContext Please see <code>getDataContext</code>
   * \param context The context beeing specified
   * \param index Index of map, if no name is specified
   * \return true if context was found
   */
  bool getVoxelmapContext(VoxelmapContext* context, uint32_t index = 0);

  /*!
   * \brief getVoxelmapContext Please see <code>getDataContext</code>
   * \param context The context beeing specified
   * \param index Index of map, if no name is specified
   * \return true if context was found
   */
  bool getVoxellistContext(CubelistContext *context, uint32_t index = 0);

  /*!
   * \brief getVoxelmapContext Please see <code>getDataContext</code>
   * \param context The context beeing specified
   * \param index Index of map, if no name is specified
   * \return true if context was found
   */
  bool getOctreeContext(CubelistContext *context, uint32_t index);

  /*!
   * \brief getVoxelmapContext Please see <code>getDataContext</code>
   * \param context The context beeing specified
   * \param index Index of map, if no name is specified
   * \return true if context was found
   */
  bool getPrimitiveArrayContext(PrimitiveArrayContext* context, uint32_t index);

  /*!
   * \brief getPrimtives can be used to display a single sphere and cube as visual
   * aids. To display more of them, use a <code>PrimitiveArray</code>.
   * Tags have to be named <code>sphere</code> and <code>cuboid</code>.
   * See <code>getSphereFromXML()</code> and <code>getCuboidFromXML</code> for details.
   * \param primitives contains the cube and/or the sphere
   */
  void getPrimtives(std::vector<Primitive*>& primitives);

  /*!
   * \brief getDefaultSphere specifies a sphere with a unit radius and 16 facetes.
   * \param sphere which is defined
   */
  void getDefaultSphere(Sphere*& sphere);

  /*!
   * \brief getDefaultCuboid specifies a unit cube
   * \param cuboid which is defined
   */
  void getDefaultCuboid(Cuboid*& cuboid);

  /*!
   * \brief getMaxMem reads the maximum allowed memory used for visualizer datastructures
   * The value is given in MByte and has to be included in the <miscellaneous>-environment.
   *
   @verbatim
   <miscellaneous>
     <max_memory_usage> 0 </max_memory_usage>  <!--In MByte, 0 = no limit -->
   </miscellaneous>
   @endverbatim
   *
   * \return maximum memory in Bytes
   */
  size_t getMaxMem();

  /*!
   * \brief getMaxFps reads the maximum allowed framerate of the visualizer.
   *
   * Example:
   @verbatim
   <miscellaneous>
     <max_fps> 60 </max_fps>  <!--In MByte, 0 = no limit -->
   </miscellaneous>
   @endverbatim
   *
   * \return maximum FPS
   */
  float getMaxFps();

private:
  /////////////////////////////private functions//////////////////////////////////

  /*!
   * \brief getXYZFromXML Reads a vector from XML
   *
   * Example:
   @verbatim
    <position>
      <x> 133.053 </x>
      <y> 143.992 </y>
      <z> 48.7809 </z>
    </position>
   @endverbatim
   *
   * \param position The result
   * \param c_path Where to read the vector
   * \param default_value Result in case of failure
   * \return true if position could be read
   */
  bool getXYZFromXML(glm::vec3& position, boost::filesystem::path c_path, float default_value);

  /*!
   * \brief colorNameToVec3 Transforms a string to a RGB color
   *
   * Possible values:
   * - black
   * - white
   * - red
   * - dark red
   * - green
   * - dark green
   * - blue
   * - dark blue
   * - gray
   * - dark gray
   * - yellow
   * - dark yellow
   * - cyan
   * - dark cyan
   * - magenta
   * - dark magenta
   *
   * \param name The name of the color
   * \return The color values. Black in error case.
   */
  glm::vec3 colorNameToVec3(std::string name);

  /*!
   * \brief getColorFromXML Reads a single color from RGBA or string
   *
   * RGBA example (values between 0.0 and 1.0):
   @verbatim
   <type_1>
     <rgba>
       <r> 0.6 </r>
       <g> 1.0 </g>
       <b> 0.2 </b>
       <a> 1.0 </a>
      </rgba>
   </type_1>
   @endverbatim
   * String example (see <code>colorNameToVec3</code> for names):
   @verbatim
   <type_1>
    cyan
   </type_1>
   @endverbatim
   * \param color The defined color
   * \param c_path The XML path to the color to read
   * \return true if parameters were found
   */
  bool getColorFromXML(glm::vec4& color, boost::filesystem::path c_path);

  /*!
   * \brief getColorPairFromXML Reads two colors from XML
   *
   * Eample for two color names:
   *
   @verbatim
    <type_0>
        <color_1>  blue  </color_1>
        <color_2>  yellow  </color_2>
    </type_0>
   @endverbatim
   *
   * Example for two RGBA colors:
   *
   @verbatim
    <type_0>
      <color_1>
       <rgba>
          <r> 0.6 </r>
          <g> 1.0 </g>
          <b> 0.2 </b>
          <a> 1.0 </a>
        </rgba>
      </color_1>
      <color_2>
        <rgba>
          <r> 1.0 </r>
          <g> 0.0 </g>
          <b> 1.0 </b>
          <a> 1.0 </a>
         </rgba>
      </color_2>
    </type_0>
   @endverbatim
   *
   * \param colors The read colors
   * \param c_path Where to read the colors
   * \return true if parameters were found
   */
  bool getColorPairFromXML(colorPair& colors, boost::filesystem::path c_path);

  /*!
   * \brief getUnitScale allows to specify a conversion rate and entity
   * which is used, when Voxel information are displayed. The tag has to
   * be within the <code><miscellaneous></code>-environment.
   *
   * Example:
   @verbatim
   <miscellaneous>
     <unit_scale> 4 cm </unit_scale>
   </miscellaneous>
   @endverbatim
   * \return true, if parameter could be read
   */
  std::pair<float, std::string> getUnitScale();


  /*!
   * \brief getDataContext Reads the parameters of a whole map
   *
   * The occupancy threshold is specified for the whole map by
   * <code><occupancy_threshold> 1-255 integer </occupancy_threshold>.
   * A visualization offset (DEPRECATED) can be set by <code><offset> xyz-offset </offset>
   * The color may be specified for all Voxel meanings within the map at once by using
   * the <code><all_types></code> field.
   * Specific meanings can afterwards been specified seperately.
   *
   * Example:
   @verbatim
    <myGraspSweepVoxellist>
      <!-- only voxels with an occupancy >= occupancy_threshold will be drawn. Has to be in the range of 1 to 255 -->
      <occupancy_threshold> 1 </occupancy_threshold>
      <all_types>
        <color>  green </color>
      </all_types>
      <type_1>
        <color> blue </color>
      </type_1>
      <type_105>
        <color> blue </color>
      </type_105>
    </myGraspSweepVoxellist>
   @endverbatim
   *
   * \param context The context that should be parametrized
   * \param name Name of the map that is parametrized
   * \return true if some parameters could be read
   */
  bool getDataContext(DataContext* context, std::string name);

  /*!
   * \brief getSphereFromXML reads parameters that are used
   * to display primitive sphere lists or a single sphere
   *
   * Eample:
   @verbatim
    <defaultSphere>
      blue
      <position>
        <x> 0 </x>
        <y> 2 </y>
        <z> 0 </z>
      </position>
      <radius>2.3</radius>
      <resolution>16</resolution>
    </defaultSphere>
   @endverbatim
   * \param sphere to be configured
   * \param name of the sphere-tag
   * \return true if parameters were found
   */
  bool getSphereFromXML(Sphere*& sphere, std::string name);

  /*!
   * \brief getCuboidFromXML reads parameters that are used
   * to display voxels or a single cube
   *
   * Example:
   @verbatim
    <defaultCuboid>
      cyan
      <position>
        <x> 0 </x>
        <y> 0 </y>
        <z> 0 </z>
      </position>
      <side_length>
        <x> 1 </x>
        <y> 1 </y>
        <z> 1 </z>
      </side_length>
    </defaultCuboid>
   @endverbatim
   *
   * \param cuboid to be configured
   * \param name of the cube-tag
   * \return true if parameters were found
   */
  bool getCuboidFromXML(Cuboid*& cuboid, std::string name);

  /*!
   * \brief getCameraFromXML reads all camera parameters
   *
   * Example:
   @verbatim
   <camera>
    <!-- Free flight mode -->
    <position>
      <x> 133.053 </x>
      <y> 143.992 </y>
      <z> 48.7809 </z>
    </position>
    <horizontal_angle> 320.31 </horizontal_angle> <!-- given in Deg -->
    <vertical_angle> 2.63661 </vertical_angle> <!-- given in Deg -->
    <field_of_view> 60 </field_of_view> <!-- given in Deg -->
    <!-- Orbit mode -->
    <focus>
      <x> 100 </x>
      <y> 100 </y>
      <z> 0 </z>
    </focus>
    <window_width> 1024 </window_width>
    <window_height> 768 </window_height>
   </camera>
   @endverbatim
   * \return The camera config object
   */
  Camera_gpu* getCameraFromXML();

  /////////////////////////////private variables/////////////////////////////////////

};

} // end of ns
} // end of ns

#endif
