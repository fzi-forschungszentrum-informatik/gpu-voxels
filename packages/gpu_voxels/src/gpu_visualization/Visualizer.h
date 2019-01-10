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
#if defined(__CUDACC__) && !defined(CUDA_VERSION) && !defined(GLM_FORCE_CUDA) // fix Cuda10 & Ubuntu14.04 error
#  include <cuda.h>  // ensure CUDA_VERSION is defined, nvcc does not define it
#endif
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
#include <gpu_visualization/SharedMemoryManagerVoxelLists.h>
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

enum MENU_ITEM
{
  MENU_HELP,
  MENU_TEXT_ALL,
  MENU_TEXT_POINTS,
  MENU_TEXT_VBO,
  MENU_TEXT_VOXELMAPS,
  MENU_TEXT_VOXELLISTS,
  MENU_TEXT_OCTREES,
  MENU_TEXT_PRIMITIVEARRAYS,
  MENU_TEXT_TYPES,
  MENU_TEXT_CLICKEDVOXELINFO,
  MENU_CAMERA_RESET,
  MENU_CAMERA_FREE,
  MENU_CAMERA_ORBIT,
  MENU_CAMERA_TOGGLETEXT,
  MENU_GRID_ON,
  MENU_GRID_OFF,
  MENU_RENDERMODE_SOLID,
  MENU_RENDERMODE_WIREFRAME,
  MENU_RENDERMODE_SOLIDWIREFRAME,
  MENU_RENDERMODE_DIST_DEFAULT,
  MENU_RENDERMODE_DIST_TWOCOLOR_GRADIENT,
  MENU_RENDERMODE_DIST_MULTICOLOR_GRADIENT,
  MENU_RENDERMODE_DIST_VORONOI_LINEAR,
  MENU_RENDERMODE_DIST_VORONOI_SCRAMBLE,
  MENU_RENDERMODE_SLICING_OFF,
  MENU_RENDERMODE_SLICING_X,
  MENU_RENDERMODE_SLICING_Y,
  MENU_RENDERMODE_SLICING_Z,
  MENU_DRAWMAP_ALL,
  MENU_DRAWMAP_VIEW,
  MENU_DEPTHTEST_ALWAYS,
  MENU_DEPTHTEST_LEQUAL,
  MENU_LIGHT_ON,
  MENU_LIGHT_OFF,
  MENU_VISIBILITYTRIGGER_ACTIVATED,
  MENU_VISIBILITYTRIGGER_DEACTIVATED,
  MENU_NONSENSE
};

class Visualizer
{
public:

  Visualizer();
  ~Visualizer();

  bool initializeVisualizer(int& argc, char *argv[]);
  void initializeDrawTextFlags();

  //callbacks
  void timerFunction(int32_t value, void (*callback)(int32_t));
  void resizeFunction(int32_t width, int32_t height);
  void renderFunction(void);
  void idleFunction(void) const;
  void cleanupFunction(void);
  void menuFunction(int);
  
  //documentation of GPU_Voxels uses this
  /**
  * <ul>
  * <li>h: Prints help.</li>
  * <li>a: move slice axis negative</li>
  * <li>b: print number of Voxels drawn.</li>
  * <li>c: Toggles between orbit and free-flight camera.</li>
  * <li>d: Toggles drawing of the grid.</li>
  * <li>e: Toggle through drawing modes for triangles </li>
  * <li>g: Toggles draw whole map. This disables the clipping of the field of view.</li>
  * <li>i: Toggles the OpenGL depth function for the collision type to draw colliding Voxles over all other Voxels. GL_ALWAYS should be used to get an overview (produces artifacts).</li>
  * <li>k: Toggles use of camera target point from shared memory (Currently not implemented).</li>
  * <li>l or x: Toggles lighting on/off.</li>
  * <li>m: Prints total VBO size in GPU memory.</li>
  * <li>n: Prints device memory info.</li>
  * <li>o: Overwrite providers possibility to trigger visibility of swept volumes: The provider may select, which Swept-Volumes are visible. This option overwrites the behaviour.</li>
  * <li>p: Print camera position. Output can directly be pasted into an XML Configfile.</li>
  * <li>q: move slice axis positive</li>
  * <li>r: Reset camera to default position.</li>
  * <li>s: Draw all swept volume types on/off (All SweptVol types will be deactivated after switching off.)</li>
  * <li>t: rotate slice axis</li>
  * <li>ALT-t: Cycles through various view modes of Distance Fields (press 2x ‘s’ afterwards to update the view, if stuck)</li>
  * <li>v: Prints view info.</li>
  * <li>+/-: Increase/decrease the light intensity. Can be multiplied with SHIFT / CTRL.</li>
  * <li>CRTL: Hold down for high movement speed.</li>
  * <li>SHIFT: Hold down for medium movement speed.</li>
  * <li>0-9: Toggles the drawing of the different Voxel-types. Pressing ALT, you can set a decimal prefix (10, 20, 30 ...)</li>
  * <li>,/.: previous/next keyboard mode: Voxelmap > Voxellist > Octree > Primitivearrays >
  * <li>F1-F11 Toggle drawing according to the keyboard mode.</li>
  * </ul>
  */
  void keyboardFunction(unsigned char, int32_t, int32_t);
  void keyboardSpecialFunction(int32_t key, int32_t x, int32_t y);
  void mouseMotionFunction(int32_t xpos, int32_t ypos);
  void mousePassiveMotionFunction(int32_t xpos, int32_t ypos);
  /**
  * <ul>
  * <li>RIGHT_BUTTON: Prints x,y,z coordinates of the clicked voxel on console.</li>
  * <li>LEFT_BUTTON: Enables mouse movement.</li>
  * <li>ALT + CTRL + LEFT_BUTTON: Enables focus point movement in X-Y-Plane for Orbit mode.</li>
  * <li>ALT + CTRL + MIDDLE_BUTTON: Enables focus point movement in Z-Direction for Orbit mode.</li>
  * <li>ALT + CTRL + MOUSE_WHEEL: Move Camera closer of further away from focus point.</li>
  * <li>MOUSE_WHEEL: Increase/ decrease super voxel size. This influences rendering performance.</li>
  * </ul>
  */
  void mouseClickFunction(int32_t button, int32_t state, int32_t x, int32_t y);

  //Data handling functions
  void registerVoxelMap(voxelmap::AbstractVoxelMap* map, uint32_t index, std::string map_name);
  void registerVoxelList(uint32_t index, std::string map_name);
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

  std::vector<std::string> getVoxelMapNames();
  std::vector<std::string> getVoxelListNames();
  std::vector<std::string> getOctreeNames();
  std::vector<std::string> getPrimitiveArrayNames();
private:
////////////////////////////////////////private functions//////////////////////////////////////////

  /**
   * Initializes OpenGL and complies the shader programs.
   */
  bool initGL(int32_t *argc, char **argv);
  /**
   * Initializes the visualizer context 
   */
  bool initializeContextFromXML();

  void deleteGLBuffer(DataContext* con);
  void generateGLBufferForDataContext(DataContext* con);

  /**
   * Resizes the VBO of the context to new_size_byte.
   */
  void resizeGLBuffer(DataContext* con, size_t new_size_byte);
  /**
   * Calculates the new necessary size for the context and resizes it if necessary.
   */
  bool resizeGLBufferForCubelist(CubelistContext* con);
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
  void fillGLBufferWithCubelist(CubelistContext* context, uint32_t index);

  void printVBO(DataContext* context);

  void calculateNumberOfCubeTypes(CubelistContext *context);

  void updateStartEndViewVoxelIndices();
  bool updateOctreeContext(CubelistContext *context, uint32_t index);
  bool updateVoxelListContext(CubelistContext *context, uint32_t index);

  void increaseSuperVoxelSize();
  void decreaseSuperVoxelSize();

  void renderFrameWithColorMap(uint32_t, DataContext *);
  uint32_t getDataPositionFromColorMap(int32_t x, int32_t y, uint32_t axis, DataContext * con);

  void distributeMaxMemory();

  void flipDrawVoxelmap(uint32_t index);
  void flipDrawOctree(uint32_t index);
  void flipDrawVoxellist(uint32_t index);
  void flipDrawPrimitiveArray(uint32_t index);
  void flipDrawType(BitVoxelMeaning type);
  void flipDrawSweptVolume();
  void flipExternalVisibilityTrigger();
  void copyDrawTypeToDevice();
  void updateTypesSegmentMapping(DataContext* context);
  __inline__ uint8_t typeToColorIndex(uint8_t type);

  void toggleLighting();

  void printHelp();
  std::string printViewInfo();
  std::string printNumberOfVoxelsDrawn();
  std::string printTotalVBOsizes();
  std::string printPositionOfVoxelUnderMouseCursor(int32_t xpos, int32_t ypos);
  void log();
  void logCreate();

  void rotate_slice_axis();
  void move_slice_axis(int offset);


  //keyboard helper
  void keyboardFlipVisibility(glm::int8_t index);
  void nextKeyboardMode();
  void lastKeyboardMode();
  std::string keyboardModetoString(int8_t index);
  void keyboardDrawTriangles();

  /**
    * Draws one line of text on the screen. (0,0) is the lower left corner of the screen.
    */
  void drawTextLine(std::string text, int x, int y);

  /**
    * Draws multiple lines of text divided by a \n on the screen. (0,0) is the lower left corner of the screen.
    */
  void drawText(std::string, int x, int y);
/////////////////////////////////////////member variables//////////////////////////////////////////
  // the current context of the visualizer
  VisualizerContext* m_cur_context;

  // if enabled, provider programs can trigger which maps should be drawn
  bool m_use_external_draw_type_triggers;

  // if enabled, the full swept volumes are drawn. If disabled start, 1, 2, 3, 4 can be triggered by number keys
  bool m_draw_swept_volumes;

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

  // enable focus point movement
  bool m_move_focus_enabled;
  bool m_move_focus_vertical_enabled;

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
  SharedMemoryManagerVoxelLists* m_shm_manager_voxellists;
  SharedMemoryManagerPrimitiveArrays* m_shm_manager_primitive_arrays;
  SharedMemoryManagerVisualizer* m_shm_manager_visualizer;

  Primitive* m_default_prim;

  std::vector<Primitive*> m_primitives;

  int8_t m_keyboardmode;
  u_int8_t m_trianglemode;

  //benchmark variables
  uint32_t m_cur_fps;
  std::string m_log_file;

  //Draw Text Flags
  bool m_drawTextAll;
  bool m_drawPointCountText;
  bool m_drawVBOText;
  bool m_drawVoxelMapText;
  bool m_drawVoxelListText;
  bool m_drawOctreeText;
  bool m_drawPrimitiveArrayText;
  bool m_drawTypeText;
  bool m_drawClickedVoxelInfo;
  bool m_drawCameraInfo;

  std::string m_clickedVoxelInfo;
};

} // end of namespace visualization
} // end of namespace gpu_voxels

#endif
