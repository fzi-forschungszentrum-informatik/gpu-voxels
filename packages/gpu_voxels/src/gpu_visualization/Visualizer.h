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
 * \date    2013-11-6
 *
 *  \brief Visualization for VoxelMaps and Octrees
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_VISUALIZATION_VISUALIZER_H_INCLUDED
#define GPU_VOXELS_VISUALIZATION_VISUALIZER_H_INCLUDED

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <time.h>
#include <assert.h>
#include <vector_types.h>
#include <string>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/thread/thread.hpp>

#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <thrust/scan.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <gpu_visualization/logging/logging_visualization.h>

#include <icl_core_config/Config.h>

#include <gpu_voxels/helpers/cuda_handling.h>
#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/voxelmap/VoxelMap.h>
#include <gpu_voxels/primitive_array/PrimitiveArray.h>
#include <gpu_visualization/kernels/VoxelMapVisualizerOperations.h>
#include <gpu_visualization/Camera.h>

#include <gpu_visualization/VisualizerContext.h>
#include <gpu_visualization/XMLInterpreter.h>

#include <gpu_visualization/Utils.h>
#include <gpu_visualization/shader.h>
#include <gpu_visualization/Cuboid.h>
#include <gpu_visualization/Sphere.h>

#include <gpu_visualization/visualizerDefines.h>

#include <gpu_visualization/SharedMemoryManagerOctrees.h>
#include <gpu_visualization/SharedMemoryManagerVoxelMaps.h>
#include <gpu_visualization/SharedMemoryManagerPrimitiveArrays.h>
#include <gpu_visualization/SharedMemoryManagerVisualizer.h>

// todo remove this
static const bool USE_BIT_VOXEL_MAP = false;

/**
 * @namespace gpu_voxels::visualization
 * Shared Memeory Visualization for VoxelMaps and Octrees
 */
namespace gpu_voxels {
namespace visualization {

class Visualizer
{
public:

  Visualizer();
  ~Visualizer();

  bool initalizeVisualizer(int& argc, char *argv[]);

  //callbacks
  void timerFunction(int32_t value, void (*callback)(int32_t));
  void resizeFunction(int32_t width, int32_t height);
  void renderFunction(void);
  void idleFunction(void) const;
  void cleanupFunction(void);
  void keyboardFunction(unsigned char, int32_t, int32_t);
  void keyboardSpecialFunction(int32_t key, int32_t x, int32_t y);
  void mouseMotionFunction(int32_t xpos, int32_t ypos);
  void mousePassiveMotionFunction(int32_t xpos, int32_t ypos);
  void mouseClickFunction(int32_t button, int32_t state, int32_t x, int32_t y);

  //Data handling functions
  void registerVoxelMap(voxelmap::AbstractVoxelMap* map, uint32_t index, std::string map_name);
  void registerOctree(uint32_t index, std::string map_name);
  void registerPrimitiveArray(uint32_t index, std::string prim_array_name);

  ////////////////////////////////Getter & Setter////////////////////////////////
  int32_t getHeight() const
  {
    return m_cur_context->m_camera->getWindowHeight();
  }

  void setHeight(int32_t height)
  {
    m_cur_context->m_camera->setWindowHeight(height);
  }

  int32_t getWidth() const
  {
    return m_cur_context->m_camera->getWindowWidth();
  }

  void setWidth(int32_t width)
  {
    m_cur_context->m_camera->setWindowWidth(width);
  }

  uint32_t getMaxMem() const
  {
    return m_max_mem;
  }

  void setMaxMem(uint32_t maxMem)
  {
    m_max_mem = maxMem;
  }

private:
////////////////////////////////////////private functions//////////////////////////////////////////

  /**
   * Initializes OpenGL and complies the shader programs.
   */
  bool initGL(int32_t *argc, char **argv);
  /**
   * Initializes the visualizer context with the parameter from the config xml file
   */
  bool initializeContextFromXML(int& argc, char *argv[]);

  void deleteGLBuffer(DataContext* con);
  void generateGLBufferForDataContext(DataContext* con);

  /**
   * Resizes the VBO of the context to new_size_byte.
   */
  void resizeGLBuffer(DataContext* con, size_t new_size_byte);
  /**
   * Calculates the new necessary size for the context and resizes it if necessary.
   */
  bool resizeGLBufferForOctree(OctreeContext* con);
  /**
    * Calculates the new necessary size for the context and resizes it.
    */
  void resizeGLBufferForWithoutPrecounting(VoxelmapContext* con);

  void createGridVBO();
  void createFocusPointVBO();
  void createHelperSphere(glm::vec3 pos, float radius, glm::vec4 color);
  void createHelperCuboid(glm::vec3 pos, glm::vec3 side_length, glm::vec4 color);

  void drawDataContext(DataContext* context);
  void drawGrid();
  void drawFocusPoint();
  void drawHelperPrimitives();
  void drawPrimitivesFromSharedMem();

  bool fillGLBufferWithoutPrecounting(VoxelmapContext* context);
  void fillGLBufferWithOctree(OctreeContext* context, uint32_t index);

  void printVBO(DataContext* context);
  void calculateNumberOfCubeTypes(OctreeContext* context);

  void updateStartEndViewVoxelIndices();
  bool updateOctreeContext(OctreeContext* context, uint32_t index);

  void increaseSuperVoxelSize();
  void decreaseSuperVoxelSize();

  void renderFrameWithColorMap(uint32_t, DataContext *);
  uint32_t getDataPositionFromColorMap(int32_t x, int32_t y, uint32_t axis, DataContext * con);

  void distributeMaxMemory();

  void flipDrawVoxelmap(uint32_t index);
  void flipDrawOctree(uint32_t index);
  void flipDrawType(VoxelType type);
  void flipExternalVisibilityTrigger();
  void copyDrawTypeToDevice();
  void updateTypesSegmentMapping(DataContext* context);
  __inline__ uint8_t typeToColorIndex(uint8_t type);

  void toggleLighting();

  void printHelp();
  void printViewInfo();
  void printNumberOfVoxelsDrawn();
  void printTotalVBOsizes();
  void printPositionOfVoxelUnderMouseCurser(int32_t xpos, int32_t ypos);
  void log();
  void logCreate();
/////////////////////////////////////////member variables//////////////////////////////////////////
  // the current context of the visualizer
  VisualizerContext* m_cur_context;

  // if enabled, provider programs can trigger which maps should be drawn
  bool m_use_external_draw_type_triggers;

  // the title of the visualizer window
  std::string m_window_title;

  // the maximum amount of bytes the visualizer may use. 0 is equal to no limitation.
  size_t m_max_mem;
  size_t m_cur_mem;
  uint32_t m_frameCount;
  float m_delta_time;
  float m_max_fps;

  // Variables for OpenGL
  GLuint m_programID;
  GLuint m_vpID;
  GLuint m_startColorID;
  GLuint m_endColorID;
  GLuint m_interpolationID;
  GLuint m_interpolationLengthID;
  GLuint m_translationOffsetID;

  GLuint m_colormap_programID;
  GLuint m_vertex_arrayID;

  // lighting variables for OpenGl
  GLuint m_lighting_programID;
  GLuint m_light_vpID;
  GLuint m_light_vID;
  GLuint m_light_v_inv_transID;
  GLuint m_light_light_intensity;
  GLuint m_light_lightposID;
  GLuint m_light_startColorID;
  GLuint m_light_endColorID;
  GLuint m_light_interpolationID;
  GLuint m_light_interpolationLengthID;
  GLuint m_light_translationOffsetID;

  XMLInterpreter* m_interpreter;
  SharedMemoryManagerOctrees* m_shm_manager_octrees;
  SharedMemoryManagerVoxelMaps* m_shm_manager_voxelmaps;
  SharedMemoryManagerPrimitiveArrays* m_shm_manager_primitive_arrays;
  SharedMemoryManagerVisualizer* m_shm_manager_visualizer;

  Primitive* m_default_prim;

  std::vector<Primitive*> m_primitives;

  //benchmark variables
  uint32_t m_cur_fps;
  std::string m_log_file;
};

} // end of namespace visualization
} // end of namespace gpu_voxels

#endif
