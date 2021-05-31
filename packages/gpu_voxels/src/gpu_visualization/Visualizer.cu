// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
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
#include "Visualizer.h"

#include <boost/interprocess/managed_shared_memory.hpp>
#include <gpu_visualization/shaders/ColormapFragmentShader.h>
#include <gpu_visualization/shaders/ColormapVertexShader.h>
#include <gpu_visualization/shaders/LightingFragmentShader.h>
#include <gpu_visualization/shaders/LightingVertexShader.h>
#include <gpu_visualization/shaders/SimpleFragmentShader.h>
#include <gpu_visualization/shaders/SimpleVertexShader.h>
#include <gpu_voxels/voxelmap/kernels/VoxelMapOperations.h>

#include <ctime>

namespace gpu_voxels {
namespace visualization {

namespace bfs = boost::filesystem;
using namespace glm;

Visualizer::Visualizer()
{
  setMaxMem(0);
  m_cur_mem = 0;
  m_default_prim = NULL;
  m_shm_manager_octrees = NULL;
  m_shm_manager_voxelmaps = NULL;
  m_shm_manager_voxellists = NULL;
  m_shm_manager_primitive_arrays = NULL;
  m_shm_manager_visualizer = NULL;
  m_window_title = "GPU-Voxels Visualizer";
  m_use_external_draw_type_triggers = false;
  m_draw_swept_volumes = true;
  m_move_focus_enabled = false;
  m_move_focus_vertical_enabled = false;
  m_keyboardmode = 0;
  m_trianglemode = 0;
  m_drawTextAll = true;
  m_drawPointCountText = false;
  m_drawVBOText = false;
  m_drawVoxelMapText = true;
  m_drawVoxelListText = true;
  m_drawOctreeText = true;
  m_drawPrimitiveArrayText = true;
  m_drawTypeText = false;
  m_drawClickedVoxelInfo = false;
  m_drawCameraInfo = false;
  m_clickedVoxelInfo = "no Voxel clicked yet";
}

Visualizer::~Visualizer()
{
  delete m_cur_context;
  delete m_interpreter;
  delete m_default_prim;
  delete m_shm_manager_octrees;
  delete m_shm_manager_voxelmaps;
  delete m_shm_manager_primitive_arrays;
  delete m_shm_manager_visualizer;
  for (std::vector<Primitive*>::iterator it = m_primitives.begin(); it != m_primitives.end(); ++it)
  {
    delete *it;
  }
}

std::vector<std::string> Visualizer::getVoxelMapNames()
{
  std::vector<std::string> names;
  for(size_t i = 0; i < m_cur_context->m_voxel_maps.size(); i++)
  {
    names.push_back(m_cur_context->m_voxel_maps[i]->m_map_name);
  }
  return names;
}

std::vector<std::string> Visualizer::getVoxelListNames()
{
  std::vector<std::string> names;
  for(size_t i = 0; i < m_cur_context->m_voxel_lists.size(); i++)
  {
    names.push_back(m_cur_context->m_voxel_lists[i]->m_map_name);
  }
  return names;
}

std::vector<std::string> Visualizer::getOctreeNames()
{
  std::vector<std::string> names;
  for(size_t i = 0; i < m_cur_context->m_octrees.size(); i++)
  {
    names.push_back(m_cur_context->m_octrees[i]->m_map_name);
  }
  return names;
}

std::vector<std::string> Visualizer::getPrimitiveArrayNames()
{
  std::vector<std::string> names;
  for(size_t i = 0; i < m_cur_context->m_prim_arrays.size(); i++)
  {
    names.push_back(m_cur_context->m_prim_arrays[i]->m_map_name);
  }
  return names;
}

bool Visualizer::initGL(int32_t* argc, char**argv)
{
  ///////////// cuda device check///////////////////////////
  int32_t dev_ID = 0;
  cudaDeviceProp deviceProp;
  HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, dev_ID));
  if (deviceProp.major < 2)
  {
    LOGGING_ERROR_C(Visualization, Visualizer, "GPU must at least support compute capability 2.x" << endl);
    return false;
  }
  LOGGING_INFO_C(
      Visualization,
      Visualizer,
      "Using GPU Device " << dev_ID << ": \"" << deviceProp.name << "\" with compute capability " << deviceProp.major << "." << deviceProp.minor << endl);

/////////// Initialize glut ///////////////////
  glutInit(argc, argv);
  //glutInitContextVersion(4, 3);
  // removes deprecated functions
  glutInitContextFlags(GLUT_FORWARD_COMPATIBLE);
  glutInitContextProfile(GLUT_CORE_PROFILE);
  glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
  glutInitWindowSize(getWidth(), getHeight());
  glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);

  if (glutCreateWindow(m_window_title.c_str()) < 1)
  {
    LOGGING_ERROR_C(Visualization, Visualizer, "Glut could not create a new rendering window." << endl);
    exit(EXIT_FAILURE);
  }
  glewExperimental = GL_TRUE;
  /////////// Initialize glew ///////////////////
  GLenum glewInitResult = glewInit();

  if (GLEW_OK != glewInitResult)
  {
    LOGGING_ERROR_C(Visualization, Visualizer,
                    " Couldn't initialize GLEW " << (char*) glewGetErrorString(glewInitResult) << endl);
    exit(EXIT_FAILURE);
  }
  //discard error, which is created from glewInit() even though it was successful.
  glGetError();
  LOGGING_INFO_C(Visualization, Visualizer,
                 "Using OpenGL Version: " << (char*) glGetString(GL_VERSION) << endl);

  //set background color
  glm::vec4 bc = m_cur_context->m_background_color;
  glClearColor(bc.x, bc.y, bc.z, 1.f);
  LOGGING_DEBUG_C(Visualization, Visualizer,
                  "Setting the background color to (" << bc.x << ", " << bc.y << ", " << bc.z << ")" << endl);

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
  ExitOnGLError("Could not set OpenGL depth testing options");

  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);
  glFrontFace(GL_CCW);
  ExitOnGLError("Could not set OpenGL culling options");

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  ExitOnGLError("Could not set OpenGL blend options");

  // Create and compile the GLSL program from the shaders
  m_programID = loadShaders(SimpleVertexShader::get(), SimpleFragmentShader::get());
  m_colormap_programID = loadShaders(ColormapVertexShader::get(), ColormapFragmentShader::get());
  m_lighting_programID = loadShaders(LightingVertexShader::get(), LightingFragmentShader::get());

  // get the positions of the uniform variables of the lighting shaders
  m_light_vpID = glGetUniformLocation(m_lighting_programID, "VP");
  m_light_vID = glGetUniformLocation(m_lighting_programID, "V");
  m_light_v_inv_transID = glGetUniformLocation(m_lighting_programID, "V_inverse_transpose");
  m_light_lightposID = glGetUniformLocation(m_lighting_programID, "lightPosition_worldspace");
  m_light_light_intensity = glGetUniformLocation(m_lighting_programID, "lightIntensity");

  m_light_startColorID = glGetUniformLocation(m_lighting_programID, "startColor");
  m_light_endColorID = glGetUniformLocation(m_lighting_programID, "endColor");
  m_light_interpolationID = glGetUniformLocation(m_lighting_programID, "interpolation");
  m_light_interpolationLengthID = glGetUniformLocation(m_lighting_programID, "interpolationLength");
  m_light_translationOffsetID = glGetUniformLocation(m_lighting_programID, "translationOffset");

  // get the positions of the uniform variables of the "simple" shaders
  m_vpID = glGetUniformLocation(m_programID, "VP");
  m_startColorID = glGetUniformLocation(m_programID, "startColor");
  m_endColorID = glGetUniformLocation(m_programID, "endColor");
  m_interpolationID = glGetUniformLocation(m_programID, "interpolation");
  m_interpolationLengthID = glGetUniformLocation(m_programID, "interpolationLength");
  m_translationOffsetID = glGetUniformLocation(m_programID, "translationOffset");

  glGenVertexArrays(1, &m_vertex_arrayID);
  glBindVertexArray(m_vertex_arrayID);

  createGridVBO();
  createFocusPointVBO();

  for (uint32_t i = 0; i < m_primitives.size(); i++)
  {
    m_primitives[i]->create(m_cur_context->m_lighting);
  }

  return true;
}

bool Visualizer::initializeContextFromXML()
{
  m_interpreter = new XMLInterpreter();

  m_max_mem = m_interpreter->getMaxMem();
  m_max_fps = m_interpreter->getMaxFps();
  m_interpreter->getPrimtives(m_primitives);

  m_cur_context = new VisualizerContext();
  bool suc = m_interpreter->getVisualizerContext(m_cur_context);
  copyDrawTypeToDevice();
  return suc;
}

bool Visualizer::initializeVisualizer(int& argc, char *argv[])
{
  LOGGING_INFO_C(Visualization, Visualizer, "Trying to open the Visualizer shared memory segment created by a process using VisProvider." << endl);
  time_t total_wait = 30; // total time used for trying to open shared memory file (seconds)
  size_t period = 500; // time to sleep between consecutive tries (milliseconds)

  time_t start = time(0);
  bool success = false;
  while (!success && time(0) < start + total_wait)
  {
    try
    {
      m_shm_manager_visualizer = new SharedMemoryManagerVisualizer();
      success = true;
    } catch (boost::interprocess::interprocess_exception& e)
    {
      usleep(period * 1000);
    }
  }
  if (!success)
  {
    m_shm_manager_visualizer = NULL;
    LOGGING_WARNING_C(Visualization, Visualizer, "Couldn't open the shared memory segment of Visualizer!" << endl);
  }

  return initializeContextFromXML() & initGL(&argc, argv);
}

void Visualizer::initializeDrawTextFlags()
{
  if(m_cur_context->m_voxel_maps.size() < 1)
  m_drawVoxelMapText = false;
  if(m_cur_context->m_voxel_lists.size() < 1)
  m_drawVoxelListText = false;  
  if(m_cur_context->m_octrees.size() < 1)
  m_drawOctreeText = false;
  if(m_cur_context->m_prim_arrays.size() < 1)
  m_drawPrimitiveArrayText = false;
}

/*!
 * Generates an OpenGL buffer for the VoxelmapContext.
 * context->vbo will contain the OpenGL buffer identifier.
 * context->cuda_ressources will contain the cudaGraphicsResource where the vbo is registered.
 */
void Visualizer::generateGLBufferForDataContext(DataContext* con)
{
  /*Reserves space for 10 voxels per voxel type.*/
  uint32_t num_voxels = 10;
  uint32_t default_size = con->m_num_voxels_per_type.size() * num_voxels * SIZE_OF_TRANSLATION_VECTOR;
  //generate the buffer
  GLuint vbo;
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, default_size, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  // register the buffer in a cudaGraphicsResource
  struct cudaGraphicsResource* res;
  HANDLE_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&res, vbo, cudaGraphicsRegisterFlagsWriteDiscard));
  ExitOnGLError("ERROR: Could not generate a new buffer");
  //fill the context
  con->m_vbo = vbo;
  con->m_cur_vbo_size = default_size;
  con->m_vbo_segment_voxel_capacities = thrust::host_vector<uint32_t>(
      con->m_vbo_segment_voxel_capacities.size(), num_voxels);

  con->m_d_vbo_segment_voxel_capacities = con->m_vbo_segment_voxel_capacities;
  con->updateVBOOffsets();
  con->m_cuda_ressources = res;
  m_cur_mem += default_size;

}

void Visualizer::deleteGLBuffer(DataContext* con)
{
  glDeleteBuffers(1, &con->m_vbo);
  con->m_vbo = 0;
}

/**
 * Calculates the new size of the VBO.
 * Returns true if the buffer needs to be resized, otherwise false.
 * If the buffer needs to be resized, /p new_size will contain the new size for the buffer.
 * @parm new_size: the requested size for the buffer, will contain the new size of the buffer afterwards.
 * @parm cur_size: the current size of the buffer, which shell be resized.

 */
bool calcSize(size_t& new_size_byte, size_t current_size)
{
  size_t req_size = std::max((size_t) 100, new_size_byte);
  size_t cur_size = current_size;

  if (req_size <= cur_size && req_size > cur_size / 4)
  {
    // no resize for the buffer needed.
    return false;
  }
  new_size_byte = req_size * (1 + BUFFER_SIZE_FACTOR);
  return true;
}

/**
 * Resizes the OpenGL buffer of the context.
 * The buffer will only be resized if necessary.
 * @return:     Returns true if the buffer can now hold all the data
 *              Returns false if the new size would exceed the memory limit.
 */
bool Visualizer::resizeGLBufferForCubelist(CubelistContext* con)
{
  size_t new_size_byte = con->getSizeForBuffer();
  if (calcSize(new_size_byte, con->m_cur_vbo_size))
  {
    if (con->m_max_vbo_size != 0 && new_size_byte > con->m_max_vbo_size)
    { // if memory limit is active and the new size would be greater than the maximum size of the vbo..
      if (con->getSizeForBuffer() <= con->m_max_vbo_size)
      { // try if the requested size without BUFFER_SIZE_FACTOR would fit.
        new_size_byte = con->getSizeForBuffer();
      }
      else
      { // the buffer cannt hold all the data, because of the memory limit.
        LOGGING_WARNING_C(Visualization, Visualizer,
                          "Increasing the super voxel size. Not enough memory!!!" << endl);
        increaseSuperVoxelSize();
        return false;
      }
    }
    ////////////////////////////////////
    resizeGLBuffer(con, new_size_byte);
    //  std::cout << getDeviceMemoryInfo();
    return true;
  }
  return true;
}

void Visualizer::resizeGLBufferForWithoutPrecounting(VoxelmapContext* context)
{
  context->m_d_vbo_segment_voxel_capacities = context->m_vbo_segment_voxel_capacities;
  context->updateVBOOffsets();
  size_t new_size_byte = 0;
  for (uint32_t i = 0; i < context->m_vbo_segment_voxel_capacities.size(); i++)
  {
    new_size_byte += context->m_vbo_segment_voxel_capacities[i];
  }
  new_size_byte *= SIZE_OF_TRANSLATION_VECTOR;
  LOGGING_DEBUG_C(Visualization, Visualizer,
                  "New VBO buffer size: " << new_size_byte / 1e+006 << " MByte" << endl);

  resizeGLBuffer(context, new_size_byte);
}

void Visualizer::resizeGLBuffer(DataContext* con, size_t new_size_byte)
{
  struct cudaGraphicsResource* cuda_res = con->m_cuda_ressources;

  m_cur_mem -= con->m_cur_vbo_size;
  GLuint vbo = con->m_vbo;
  //  std::cout << getDeviceMemoryInfo();
  LOGGING_DEBUG_C(Visualization, Visualizer,
                  "New buffer size: " << new_size_byte / 1e+006 << " MByte" << endl);
  assert(new_size_byte > 0);
  HANDLE_CUDA_ERROR(cudaGraphicsUnregisterResource(cuda_res));
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, new_size_byte, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  ExitOnGLError("Couldn't resize the OpenGL buffer.");
  HANDLE_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&cuda_res, vbo, cudaGraphicsRegisterFlagsWriteDiscard));
  con->m_cuda_ressources = cuda_res;
  con->m_cur_vbo_size = new_size_byte;
  m_cur_mem += new_size_byte;
}

void Visualizer::createGridVBO()
{
  float distance = m_cur_context->m_grid_distance; /*distance between the grid lines*/
  if (distance <= 0)
  {
    LOGGING_WARNING_C(Visualization, Visualizer,
                      "Grid distance must be greater zero! Using grid distance of 10 instead." << endl);
    m_cur_context->m_grid_distance = distance = 10;
  }

  float x_max = m_cur_context->m_grid_max_x;
  float y_max = m_cur_context->m_grid_max_y;

  x_max += distance - mod(x_max, distance);
  y_max += distance - mod(y_max, distance);

  std::vector<glm::vec3> data;
  for (float i = 0; i <= x_max; i += distance)
  {/*Insert the lines along the x axis*/
    glm::vec3 p1 = glm::vec3(i, 0, 0);
    glm::vec3 p2 = glm::vec3(i, y_max, 0);
    data.push_back(p1);
    data.push_back(p2);
  }
  for (float i = 0; i <= y_max; i += distance)
  {/*Insert the lines along the y axis*/
    glm::vec3 p1 = glm::vec3(0, i, 0);
    glm::vec3 p2 = glm::vec3(x_max, i, 0);
    data.push_back(p1);
    data.push_back(p2);
  }
  size_t size = 2.f * ((y_max + x_max) / distance + 2); // two points per line and one line for each x and y value
  assert(data.size() == size);
  size_t size_in_byte = size * sizeof(glm::vec3);

  glGenBuffers(1, &m_cur_context->m_grid_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, m_cur_context->m_grid_vbo);
  glBufferData(GL_ARRAY_BUFFER, size_in_byte, data.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  ExitOnGLError("Error! Couldn't generate the grid VBO.");
}

void Visualizer::createFocusPointVBO()
{
  glm::vec3 focus = m_cur_context->m_camera->getCameraTarget();
  glGenBuffers(1, &m_cur_context->m_focus_point_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, m_cur_context->m_focus_point_vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(focus), glm::value_ptr(focus), GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  ExitOnGLError("Error! Couldn't generate the focus point VBO.");
}

void Visualizer::createHelperSphere(glm::vec3 pos, float radius, glm::vec4 color)
{
  Sphere* s = new Sphere(color, pos, radius, 16);
  s->create(m_cur_context->m_lighting);
  m_primitives.push_back(s);
}

void Visualizer::createHelperCuboid(glm::vec3 pos, glm::vec3 side_length, glm::vec4 color)
{
  Cuboid* c = new Cuboid(color, pos, side_length);
  c->create(m_cur_context->m_lighting);
  m_primitives.push_back(c);

}

void Visualizer::registerOctree(uint32_t index, std::string map_name)
{
  if (m_shm_manager_octrees == NULL)
  {
    try
    {
      m_shm_manager_octrees = new SharedMemoryManagerOctrees();
    } catch (boost::interprocess::interprocess_exception& e)
    {
      LOGGING_ERROR_C(
          Visualization,
          Visualizer,
          "Registering the Octree with index " << index << " failed! Couldn't open the shared memory segment!" << endl);
      exit(EXIT_FAILURE);
    }
  }

  CubelistContext* con = new CubelistContext(map_name);
  if (!m_interpreter->getOctreeContext(con, index))
  {
    LOGGING_WARNING_C(Visualization, Visualizer,
                      "No context found for octree " << map_name << ". Using the default context." << endl);
  }
  generateGLBufferForDataContext(con);
  m_cur_context->m_octrees.push_back(con);
  distributeMaxMemory();

  m_shm_manager_octrees->setOctreeOccupancyThreshold(index, con->m_occupancy_threshold);
  uint32_t sdim;
  if (m_shm_manager_octrees->getSuperVoxelSize(sdim))
  {
    m_cur_context->m_dim_svoxel = Vector3ui(std::max((uint32_t) 1, sdim));
    LOGGING_INFO_C(
        Visualization,
        Visualizer,
        "The initial super voxel size of the octree could be loaded and will be used. Dimension of super voxel: " << sdim << endl);
  }
}

void Visualizer::registerVoxelMap(voxelmap::AbstractVoxelMap* map, uint32_t index, std::string map_name)
{
  if (m_shm_manager_voxelmaps == NULL)
  {
    try
    {
      m_shm_manager_voxelmaps = new SharedMemoryManagerVoxelMaps();
    } catch (boost::interprocess::interprocess_exception& e)
    {
      LOGGING_ERROR_C(
          Visualization,
          Visualizer,
          "Registering the Voxel Map with index " << index << " failed! Couldn't open the shared memory segment!" << endl);
      exit(EXIT_FAILURE);
    }
  }
  VoxelmapContext* con = new VoxelmapContext(map, map_name);
  if (!m_interpreter->getVoxelmapContext(con, index))
  {
    LOGGING_WARNING_C(
        Visualization, Visualizer,
        "No context found for voxel map " << map_name << ". Using the default context." << endl);
  }
  con->updateCudaLaunchVariables(m_cur_context->m_dim_svoxel);
  generateGLBufferForDataContext(con);
  Vector3ui d = m_cur_context->m_max_voxelmap_dim = maxVec(m_cur_context->m_max_voxelmap_dim,
                                                           map->getDimensions());

  glm::vec3 t = m_cur_context->m_camera->getCameraTarget();
  if (t.x == -0.000001f && t.x == t.y && t.y == t.z)
  {
    m_cur_context->m_camera->setCameraTarget(vec3(d.x / 2.f, d.y / 2.f, 0.f));
    m_cur_context->m_camera->setCameraTargetOfInitContext(vec3(d.x / 2.f, d.y / 2.f, 0.f));
    createFocusPointVBO();
  }
  m_cur_context->m_voxel_maps.push_back(con);
  distributeMaxMemory();
}

void Visualizer::registerVoxelList(uint32_t index, std::string map_name)
{
  if (m_shm_manager_voxellists == NULL)
  {
    try
    {
      m_shm_manager_voxellists = new SharedMemoryManagerVoxelLists();
    } catch (boost::interprocess::interprocess_exception& e)
    {
      LOGGING_ERROR_C(
          Visualization,
          Visualizer,
          "Registering the Voxel List with index " << index << " failed! Couldn't open the shared memory segment!" << endl);
      exit(EXIT_FAILURE);
    }
  }
  CubelistContext* con = new CubelistContext(map_name);
  if (!m_interpreter->getVoxellistContext(con, index))
  {
    LOGGING_WARNING_C(
        Visualization, Visualizer,
        "No context found for voxel list " << map_name << ". Using the default context." << endl);
  }
//  con->updateCudaLaunchVariables(m_cur_context->m_dim_svoxel);
  generateGLBufferForDataContext(con);

  m_cur_context->m_voxel_lists.push_back(con);
  distributeMaxMemory();
}

void Visualizer::registerPrimitiveArray(uint32_t index, std::string prim_array_name)
{
  if (m_shm_manager_primitive_arrays == NULL)
  {
    try
    {
      m_shm_manager_primitive_arrays = new SharedMemoryManagerPrimitiveArrays();
    } catch (boost::interprocess::interprocess_exception& e)
    {
      LOGGING_ERROR_C(
          Visualization,
          Visualizer,
          "Registering the Primitive Array with index " << index << " failed! Couldn't open the shared memory segment!" << endl);
      exit(EXIT_FAILURE);
    }
  }

  PrimitiveArrayContext* con = new PrimitiveArrayContext(prim_array_name);
  if (!m_interpreter->getPrimitiveArrayContext(con, index))
  {
    LOGGING_WARNING_C(Visualization, Visualizer,
                      "No context found for Primitive Array " << prim_array_name << ". Using the default context." << endl);
  }
  m_cur_context->m_prim_arrays.push_back(con);
  distributeMaxMemory();
}

/**
 * Fills the VBO from a VoxelmapContext with the translation_scale vectors.
 * @return: returns false if the VBO doesn't contain the whole map(view).
 */
bool Visualizer::fillGLBufferWithoutPrecounting(VoxelmapContext* context)
{
  updateStartEndViewVoxelIndices();
  thrust::device_vector<uint32_t> indices(context->m_num_voxels_per_type.size(), 0);

  float4 *vbo_ptr; //float4 because the translation (x,y,z) and cube size (w) will be stored in there
  size_t num_bytes; // size of the buffer
  HANDLE_CUDA_ERROR(cudaGraphicsMapResources(1, &(context->m_cuda_ressources), 0));
  HANDLE_CUDA_ERROR(
      cudaGraphicsResourceGetMappedPointer((void ** )&vbo_ptr, &num_bytes, context->m_cuda_ressources));

  // Launch kernel to copy data into the OpenGL buffer.
  // fill_vbo_without_precounting<<< dim3(1,1,1), dim3(1,1,1)>>>(/**/
  // CHECK_CUDA_ERROR();
  if (context->m_voxelMap->getMapType() == MT_BITVECTOR_VOXELMAP)
  {
    if(BIT_VECTOR_LENGTH > MAX_DRAW_TYPES)
      LOGGING_ERROR_C(Visualization, Visualizer,
          "Only " << MAX_DRAW_TYPES << " different draw types supported. But bit vector has " << BIT_VECTOR_LENGTH << " different types." << endl);

    fill_vbo_without_precounting<<<context->m_num_blocks, context->m_threads_per_block>>>(
        /**/
        (BitVectorVoxel*) context->m_voxelMap->getVoidDeviceDataPtr(),/**/
        context->m_voxelMap->getDimensions(),/**/
        m_cur_context->m_dim_svoxel,/**/
        m_cur_context->m_view_start_voxel_pos,/**/
        m_cur_context->m_view_end_voxel_pos,/**/
        context->m_occupancy_threshold,/**/
        vbo_ptr,/**/
        thrust::raw_pointer_cast(context->m_d_vbo_offsets.data()),/**/
        thrust::raw_pointer_cast(context->m_d_vbo_segment_voxel_capacities.data()),/**/
        thrust::raw_pointer_cast(indices.data()),/**/
        thrust::raw_pointer_cast(m_cur_context->m_d_draw_types.data()),/**/
        thrust::raw_pointer_cast(m_cur_context->m_d_prefixes.data()));/**/
    CHECK_CUDA_ERROR();

  }
  else if(context->m_voxelMap->getMapType() == MT_PROBAB_VOXELMAP)
  {
    fill_vbo_without_precounting<<<context->m_num_blocks, context->m_threads_per_block>>>(
        /**/
        (ProbabilisticVoxel*) context->m_voxelMap->getVoidDeviceDataPtr(),/**/
        context->m_voxelMap->getDimensions(),/**/
        m_cur_context->m_dim_svoxel,/**/
        m_cur_context->m_view_start_voxel_pos,/**/
        m_cur_context->m_view_end_voxel_pos,/**/
        context->m_occupancy_threshold,/**/
        vbo_ptr,/**/
        thrust::raw_pointer_cast(context->m_d_vbo_offsets.data()),/**/
        thrust::raw_pointer_cast(context->m_d_vbo_segment_voxel_capacities.data()),/**/
        thrust::raw_pointer_cast(indices.data()),/**/
        thrust::raw_pointer_cast(m_cur_context->m_d_draw_types.data()),/**/
        thrust::raw_pointer_cast(m_cur_context->m_d_prefixes.data()));/**/
    CHECK_CUDA_ERROR();
  }
  else if(context->m_voxelMap->getMapType() == MT_DISTANCE_VOXELMAP)
  {
    fill_vbo_without_precounting<<<context->m_num_blocks, context->m_threads_per_block>>>(
        /**/
        (DistanceVoxel*) context->m_voxelMap->getVoidDeviceDataPtr(),/**/
        context->m_voxelMap->getDimensions(),/**/
        m_cur_context->m_dim_svoxel,/**/
        m_cur_context->m_view_start_voxel_pos,/**/
        m_cur_context->m_view_end_voxel_pos,/**/
        static_cast<visualizer_distance_drawmodes>(m_cur_context->m_distance_drawmode),/**/
        vbo_ptr,/*TODO: if there is a way to pass GL_RGBA color info to OpenGL, generate those colors here too? would need to register and map additional cuda resource*/
        thrust::raw_pointer_cast(context->m_d_vbo_offsets.data()),/**/
        thrust::raw_pointer_cast(context->m_d_vbo_segment_voxel_capacities.data()),/**/
        thrust::raw_pointer_cast(indices.data()),/**/
        thrust::raw_pointer_cast(m_cur_context->m_d_draw_types.data()),/**/
        thrust::raw_pointer_cast(m_cur_context->m_d_prefixes.data()));/**/
    CHECK_CUDA_ERROR();
  }
  else
  {
    LOGGING_ERROR_C(Visualization, Visualizer,
        "No implementation to fill a voxel map of this type!" << endl);
    exit(EXIT_FAILURE);
  }
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  HANDLE_CUDA_ERROR(cudaGraphicsUnmapResources(1, &context->m_cuda_ressources, 0));

  context->m_num_voxels_per_type = indices;
  bool resize = false;
  bool increaseSuperVoxel = false;
  size_t vbo_size = context->m_cur_vbo_size;
  for (uint32_t i = 0; i < context->m_num_voxels_per_type.size(); i++)
  {
    uint32_t cur_size = context->m_vbo_segment_voxel_capacities[i];
    uint32_t req_size = context->m_num_voxels_per_type[i];
    size_t cur_size_byte = cur_size * SIZE_OF_TRANSLATION_VECTOR;
    size_t req_size_byte = req_size * SIZE_OF_TRANSLATION_VECTOR;

    if (req_size == 0 || (req_size <= cur_size && req_size > cur_size / 4))
    {
      // no resize for this segment needed. Continue with the next one.
      continue;
    }
    if (req_size > cur_size)
    {
      //num voxels per type contains the number of voxel to be drawn. May not be greater than the capacity
      context->m_num_voxels_per_type[i] = context->m_vbo_segment_voxel_capacities[i];

      // max vbo size is in Byte, req_size in number of voxels!!!
      if (context->m_max_vbo_size != 0
          && vbo_size - cur_size_byte + req_size_byte * (1 + BUFFER_SIZE_FACTOR) > context->m_max_vbo_size)
      { //if the req buffer size is greater than the max size ...

        if (vbo_size - cur_size_byte + req_size_byte <= context->m_max_vbo_size)
        { // but without the factor it would fit ... don't use the factor
          vbo_size -= cur_size_byte;
          vbo_size += req_size_byte;
          context->m_vbo_segment_voxel_capacities[i] = req_size;
          resize = true;
        }
        else
        {
          // the vbo can't hold all the data => increase the super voxel size to reduce data.
          increaseSuperVoxel = true;
          LOGGING_WARNING_C(Visualization, Visualizer,
                            "Increasing super voxel size. Not enough memory available!!!" << endl);
        }
      }
      else
      { // the new buffer size can be used..
        vbo_size -= cur_size_byte;
        vbo_size += req_size_byte * (1 + BUFFER_SIZE_FACTOR);
        context->m_vbo_segment_voxel_capacities[i] = req_size * (1 + BUFFER_SIZE_FACTOR);
        resize = true;
      }
    }
    else if (req_size > cur_size / 4)
    { // if the segment is 4 times greater than the requested size. Reduce the segment size.
      vbo_size -= cur_size_byte;
      vbo_size += req_size_byte * (1 + BUFFER_SIZE_FACTOR);
      context->m_vbo_segment_voxel_capacities[i] = req_size * (1 + BUFFER_SIZE_FACTOR);
      resize = true;

    }
  }
  context->updateTotalNumVoxels();
  updateTypesSegmentMapping(context);
  if (increaseSuperVoxel)
  {
    increaseSuperVoxelSize();
    context->m_vbo_draw_able = false;
    return false;
  }
  else if (resize)
  {
    resizeGLBufferForWithoutPrecounting(context);
    context->m_vbo_draw_able = true;
    return false;
  }
  else
  {
    /*if increaseSuperVoxel and resize is false
     *than the buffer was already big enough with its initial value*/
    context->m_vbo_draw_able = true;
    return true;
  }
}

/**
 * Fills the VBO from a Cubelist extracted from Voxellist or Octree with the translation_scale vectors.
 */
void Visualizer::fillGLBufferWithCubelist(CubelistContext* context, uint32_t index)
{
  thrust::device_vector<uint32_t> indices(context->m_num_voxels_per_type.size(), 0);
  calculateNumberOfCubeTypes(context);

  context->m_vbo_draw_able = resizeGLBufferForCubelist(context);
  if (context->m_vbo_draw_able)
  {
    float4 *vbo_ptr;
    size_t num_bytes;
    HANDLE_CUDA_ERROR(cudaGraphicsMapResources(1, &(context->m_cuda_ressources), 0));
    HANDLE_CUDA_ERROR(
        cudaGraphicsResourceGetMappedPointer((void ** )&vbo_ptr, &num_bytes, context->m_cuda_ressources));

    // Launch kernel to copy data into the OpenGL buffer.
    fill_vbo_with_cubelist<<<context->m_num_blocks, context->m_threads_per_block>>>(
        /**/
        context->getCubesDevicePointer(),/**/
        context->getNumberOfCubes(),/**/
        vbo_ptr,/**/
        thrust::raw_pointer_cast(context->m_d_vbo_offsets.data()),/**/
        thrust::raw_pointer_cast(indices.data()),/**/
        thrust::raw_pointer_cast(m_cur_context->m_d_draw_types.data()),/**/
        thrust::raw_pointer_cast(m_cur_context->m_d_prefixes.data()));/**/
    CHECK_CUDA_ERROR();

    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    HANDLE_CUDA_ERROR(cudaGraphicsUnmapResources(1, &context->m_cuda_ressources, 0));
    //cudaIpcCloseMemHandle(context->getCubesDevicePointer()); // moved out of this function and added to caller as it is required for Octrees but makes live hard for voxellists.
    updateTypesSegmentMapping(context);
  }
}

void Visualizer::printVBO(DataContext* context)
{
  float4 *dptr;
  size_t num_bytes;
  HANDLE_CUDA_ERROR(cudaGraphicsMapResources(1, &(context->m_cuda_ressources), 0));
  HANDLE_CUDA_ERROR(
      cudaGraphicsResourceGetMappedPointer((void ** )&dptr, &num_bytes, context->m_cuda_ressources));
  size_t num_float4s = num_bytes / sizeof(float4);

  LOGGING_DEBUG_C(Visualization, Visualizer, "Printing VBO:" << endl);

  float4* dest = (float4*) malloc(num_bytes);
  cudaMemcpy(dest, dptr, num_bytes, cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < num_float4s; i++)
  {
    LOGGING_DEBUG_C(
        Visualization, Visualizer,
        i << ":    " << dest[i].x << ", " << dest[i].y << ", " << dest[i].z << ", " << dest[i].w << endl);
  }
  free(dest);
  HANDLE_CUDA_ERROR(cudaGraphicsUnmapResources(1, &(context->m_cuda_ressources), 0));
}

void Visualizer::calculateNumberOfCubeTypes(CubelistContext* context)
{
  thrust::fill(context->m_d_num_voxels_per_type.begin(), context->m_d_num_voxels_per_type.end(), 0);
// Launch kernel to copy data into the OpenGL buffer. <<<context->getNumberOfCubes(),1>>><<<num_threads_per_block,num_blocks>>>
  calculate_cubes_per_type_list<<<context->m_num_blocks, context->m_threads_per_block>>>(
      context->getCubesDevicePointer(),/**/
      context->getNumberOfCubes(),/**/
      thrust::raw_pointer_cast(context->m_d_num_voxels_per_type.data()),
      thrust::raw_pointer_cast(m_cur_context->m_d_draw_types.data()),/**/
      thrust::raw_pointer_cast(m_cur_context->m_d_prefixes.data()));/**/
  CHECK_CUDA_ERROR();

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  context->m_num_voxels_per_type = context->m_d_num_voxels_per_type;
  context->updateVBOOffsets();
  context->updateTotalNumVoxels();
}

void Visualizer::updateStartEndViewVoxelIndices()
{
  if (m_cur_context->m_draw_whole_map)
  {
    if (m_cur_context->m_slice_axis == 0) {
      m_cur_context->m_view_start_voxel_pos = m_cur_context->m_min_xyz_to_draw;
      m_cur_context->m_view_end_voxel_pos = m_cur_context->m_max_xyz_to_draw;
      return;
    }

    if (m_cur_context->m_slice_axis == 1) {
      m_cur_context->m_view_start_voxel_pos = Vector3ui(m_cur_context->m_slice_axis_position, 0, 0);
      m_cur_context->m_view_end_voxel_pos = Vector3ui(m_cur_context->m_slice_axis_position + 1, (unsigned)-1, (unsigned)-1);
      return;
    }

    if (m_cur_context->m_slice_axis == 2) {
      m_cur_context->m_view_start_voxel_pos = Vector3ui(0, m_cur_context->m_slice_axis_position, 0);
      m_cur_context->m_view_end_voxel_pos = Vector3ui((unsigned)-1, m_cur_context->m_slice_axis_position + 1, (unsigned)-1);
      return;
    }

    if (m_cur_context->m_slice_axis == 3) {
      m_cur_context->m_view_start_voxel_pos = Vector3ui(0, 0, m_cur_context->m_slice_axis_position);
      m_cur_context->m_view_end_voxel_pos = Vector3ui((unsigned)-1, (unsigned)-1, m_cur_context->m_slice_axis_position + 1);
      return;
    }

    //should not happen
    return;
  }

// the center of the map in X-Y-Plane
  glm::vec3 map_center = glm::vec3(m_cur_context->m_max_voxelmap_dim.x / 2.f,
                                   m_cur_context->m_max_voxelmap_dim.y / 2.f, 0.f);
//  glm::vec3 map_center = m_cur_context->camera->getCameraTarget();

// the distance from the map center to the camera
  glm::vec3 cam_dir = m_cur_context->m_camera->getCameraDirection();
  cam_dir = glm::vec3(cam_dir.x, cam_dir.z, cam_dir.y); // swap y and z to use z has height
  glm::vec3 cam_pos = m_cur_context->m_camera->getCameraPosition();
  cam_pos = glm::vec3(cam_pos.x, cam_pos.z, cam_pos.y); // swap y and z to use z has height

  float distance_mapcenter_camera = glm::distance(map_center, cam_pos);

  m_cur_context->m_dim_view = glm::vec3(
      glm::max(distance_mapcenter_camera / 2.f, m_cur_context->m_min_view_dim));

  glm::vec4 shift_vec = glm::scale(m_cur_context->m_dim_view) * glm::vec4(cam_dir, 0.f);

  glm::vec3 center_of_view = glm::vec3(shift_vec) + cam_pos;

  center_of_view = glm::vec3(center_of_view.x, center_of_view.y, center_of_view.z);

  glm::vec3 start_voxel_index = glm::round((center_of_view - m_cur_context->m_dim_view)); // / convertFromVector3uiToVec3(m_dim_super_voxel));
  Vector3ui t = convertFromVec3ToVector3ui(glm::max(start_voxel_index, glm::vec3(0.f)));
  m_cur_context->m_view_start_voxel_pos = t - (t % m_cur_context->m_dim_svoxel)
      + m_cur_context->m_min_xyz_to_draw;

  glm::vec3 end_voxel_index = glm::round(center_of_view + m_cur_context->m_dim_view);
  m_cur_context->m_view_end_voxel_pos = minVec(
      m_cur_context->m_max_xyz_to_draw, convertFromVec3ToVector3ui(glm::max(end_voxel_index, glm::vec3(0.f))))
      / m_cur_context->m_dim_svoxel;
// first divide by dim supervoxel and multiple afterwards to cut off at the correct x,y,z value.
  m_cur_context->m_view_end_voxel_pos = m_cur_context->m_view_end_voxel_pos * m_cur_context->m_dim_svoxel;

  if (m_shm_manager_octrees != NULL)
  {
    m_shm_manager_octrees->setView(m_cur_context->m_view_start_voxel_pos,
                                   m_cur_context->m_view_end_voxel_pos);
  }

}

bool Visualizer::updateOctreeContext(CubelistContext* context, uint32_t index)
{
  Cube* cubes;
  uint32_t size;

  bool suc = m_shm_manager_octrees->getOctreeVisualizationData(cubes, size, index);
  if (suc)
  {
    context->setCubesDevicePointer(cubes);
    context->setNumberOfCubes(size);
    context->updateCudaLaunchVariables();
  }
  return suc;
}

bool Visualizer::updateVoxelListContext(CubelistContext* context, uint32_t index)
{
  Cube* cubes;
  uint32_t size;

  bool suc = m_shm_manager_voxellists->getVisualizationData(cubes, size, index);
  if (suc)
  {
    context->setCubesDevicePointer(cubes);
    context->setNumberOfCubes(size);
    context->updateCudaLaunchVariables();
  }
  return suc;
}

/*!
 * Renders the frame, but it will not be displayed.
 * The color interpolation takes place along the specified axis
 * @param axis: 0 <=> x-axis
 *              1 <=> y-axis
 *              2 <=> z-axis
 */
void Visualizer::renderFrameWithColorMap(uint32_t axis, DataContext * con)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glUseProgram(m_colormap_programID);

  mat4 VP = m_cur_context->m_camera->getProjectionMatrix() * m_cur_context->m_camera->getViewMatrix();

  GLuint vpID = glGetUniformLocation(m_colormap_programID, "VP");
  GLuint axisID = glGetUniformLocation(m_colormap_programID, "axis");

  glUniformMatrix4fv(vpID, 1, GL_FALSE, value_ptr(VP));
  glUniform1i(axisID, axis);
  ExitOnGLError("ERROR! Couldn't load variables to shader.");

  glVertexAttribDivisor(2, 1); // increase the translation per cube not vertex
//////////////////////////////////draw the vbo///////////////////////////
  for (uint32_t i = 0; i < con->m_num_voxels_per_type.size(); ++i)
  {
    uint32_t num_primitive_to_render = con->m_num_voxels_per_type[i];
    if (num_primitive_to_render)
    {
      // uint32_t offset = context->calculateOffset(i);
      uint32_t offset = con->getOffset(i);
      glBindBuffer(GL_ARRAY_BUFFER, con->m_vbo);
      // the buffer stores the translation vectors (at pos x,y,z)and the super voxel size (at pos w)for the cube
      glEnableVertexAttribArray(2);
      glVertexAttribPointer(2, // attribute 0 (must match the layout in the shader).
          4, // size
          GL_FLOAT, // type
          GL_FALSE, // normalized?
          0, // stride
          (uint32_t*) (offset * SIZE_OF_TRANSLATION_VECTOR) // use the translation vectors at the correct offset
          );
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
      con->m_default_prim->draw(num_primitive_to_render, m_cur_context->m_lighting);
      ExitOnGLError("ERROR! Couldn't draw the filled triangles.");
    }
  }
  glVertexAttribDivisor(2, 0);
  glDisableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glUseProgram(0);
}
/*!
 * Returns the axis position of the voxel at specified window coordinate.
 * @param c_x the x window coordinate
 * @param c_y the y window coordinate
 * @param axis: 0 <=> x-axis
 *              1 <=> y-axis
 *              2 <=> z-axis
 */
uint32_t Visualizer::getDataPositionFromColorMap(const int32_t c_x, const int32_t c_y, uint32_t axis,
                                                 DataContext * con)
{
  float pixel[] =
  { 0.f, 0.f, 0.f };
  renderFrameWithColorMap(axis, con);
  glReadPixels(c_x, m_cur_context->m_camera->getWindowHeight() - c_y, 1, 1, GL_RGB, GL_FLOAT, pixel);

  // reconstruct axis position from color tuple
  // color channel values are float [0..1]
  float v = 256.f * 256.f * pixel[0]/*r*/+ 256.f * pixel[1]/*g*/+ pixel[2]/*b*/;
  // scale color channel values back to int [0..255]
  return v * 255;
}
/*!
 * Draws all the data, which is currently hold by this context.
 *
 */
void Visualizer::drawDataContext(DataContext* context)
{
  if (!context->m_vbo_draw_able)
  {
    return;
  }

  glVertexAttribDivisor(2, 1); // increase the translation per primitive not vertex
  GLuint start_color_id, end_color_id;
  if (m_cur_context->m_lighting)
  {
    glUniform3fv(m_light_translationOffsetID, 1, glm::value_ptr(context->m_translation_offset));
    start_color_id = m_light_startColorID;
    end_color_id = m_light_endColorID;
  }
  else
  {
    glUniform3fv(m_translationOffsetID, 1, glm::value_ptr(context->m_translation_offset));
    start_color_id = m_startColorID;
    end_color_id = m_endColorID;
  }

//////////////////////////////////draw the vbo///////////////////////////
  for (uint32_t i = 0; i < context->m_num_voxels_per_type.size(); ++i)
  {
    uint32_t num_primitive_to_render = context->m_num_voxels_per_type[i];

    if (num_primitive_to_render && m_cur_context->m_draw_types[context->m_types_segment_mapping[i]])
    {
      // if collision type will be drawn set the deep function so that they will be drawn
      //over other triangles with the same z value.
      if (context->m_types_segment_mapping[i] == eBVM_COLLISION)
      {
        if (m_cur_context->m_draw_collison_depth_test_always)
          glDepthFunc(GL_ALWAYS);
        else
          glDepthFunc(GL_LEQUAL);
      }
      else
      {
        glDepthFunc(GL_LESS);
      }
      uint32_t offset = context->getOffset(i);
      glBindBuffer(GL_ARRAY_BUFFER, context->m_vbo);
      // the buffer stores the translation vectors (at pos x,y,z)and the super voxel size (at pos w)for the cube
      glEnableVertexAttribArray(2);
      glVertexAttribPointer(2, // attribute 0 (must match the layout in the shader).
          4, // size
          GL_FLOAT, // type
          GL_FALSE, // normalized?
          0, // stride
          (uint32_t*) (offset * SIZE_OF_TRANSLATION_VECTOR) // use the translation vectors at the correct offset
          );

      uint8_t color_index = typeToColorIndex(context->m_types_segment_mapping[i]);
      if (m_cur_context->m_draw_filled_triangles)
      {
        glUniform4fv(start_color_id, 1, glm::value_ptr(context->m_colors[color_index].first));
        glUniform4fv(end_color_id, 1, glm::value_ptr(context->m_colors[color_index].second));

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        context->m_default_prim->draw(num_primitive_to_render, m_cur_context->m_lighting);
        ExitOnGLError("ERROR! Couldn't draw the filled triangles.");
      }

      if (m_cur_context->m_draw_edges_of_triangels)
      {
        glPolygonOffset(-1.f, -1.f);
        glEnable(GL_POLYGON_OFFSET_LINE);
        //set color for the lines.
        glm::vec4 c1 = context->m_colors[color_index].first * 0.5f;
        glm::vec4 c2 = context->m_colors[color_index].second * 0.5f;

        glUniform4fv(start_color_id, 1, glm::value_ptr(c1));
        glUniform4fv(end_color_id, 1, glm::value_ptr(c2));

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        context->m_default_prim->draw(num_primitive_to_render, m_cur_context->m_lighting);
        glDisable(GL_POLYGON_OFFSET_LINE);
        ExitOnGLError("ERROR! Couldn't draw the edges of the triangles.");
      }
    }
  }

  glVertexAttribDivisor(2, 0);
  glDisableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Visualizer::drawGrid()
{
  if (m_cur_context->m_grid_vbo != 0)
  {/*Only draw the grid if the VBO was generated*/
    mat4 MVP = m_cur_context->m_camera->getProjectionMatrix() * m_cur_context->m_camera->getViewMatrix()
        * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.f, m_cur_context->m_grid_height));

    glUniformMatrix4fv(m_vpID, 1, GL_FALSE, glm::value_ptr(MVP));
    glUniform4fv(m_startColorID, 1, glm::value_ptr(m_cur_context->m_grid_color));
    glUniform1i(m_interpolationID, GL_FALSE);
    ExitOnGLError("ERROR! Couldn't load variables to shader.");

    // bind the vbo buffer
    glBindBuffer(GL_ARRAY_BUFFER, m_cur_context->m_grid_vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, // attribute 0 (must match the layout in the shader).
        3, // size
        GL_FLOAT, // type
        GL_FALSE, // normalized?
        0, // stride
        (void*) 0 // array buffer offset
        );
    ExitOnGLError("ERROR: Couldn't set the vertex attribute pointer.");

    float distance = m_cur_context->m_grid_distance; /*distance between the grid lines*/

//    float x_max = m_cur_context->max_voxelmap_dim.x;
//    float y_max = m_cur_context->max_voxelmap_dim.y;
    float x_max = m_cur_context->m_grid_max_x;
    float y_max = m_cur_context->m_grid_max_y;

    x_max += distance - mod(x_max, distance);
    y_max += distance - mod(y_max, distance);

    size_t number_line_points = 2.f * ((y_max + x_max) / distance + 2); // two points per line and one line for each x and z value

    glEnable(GL_LINE_SMOOTH);
    glUniform4f(m_startColorID, 0, 1, 0, 1); // draw the Y-axis with green
    glDrawArrays(GL_LINES, 0, 2); // << == this draws the Y axis
    glUniform4f(m_startColorID, 1, 0, 0, 1); // draw the x-axis with red
    glDrawArrays(GL_LINES, 2 * (x_max / distance + 1), 2);
    glUniform4fv(m_startColorID, 1, glm::value_ptr(m_cur_context->m_grid_color));
    glDrawArrays(GL_LINES, 2, number_line_points);

    glDisable(GL_LINE_SMOOTH);
    ExitOnGLError("ERROR! Couldn't draw the grid.");
  }
}

void Visualizer::drawFocusPoint()
{
  if (m_cur_context->m_focus_point_vbo != 0)
  {/*Only draw the focus point if the VBO was generated*/
    mat4 MVP = m_cur_context->m_camera->getProjectionMatrix() * m_cur_context->m_camera->getViewMatrix();
    glUniformMatrix4fv(m_vpID, 1, GL_FALSE, glm::value_ptr(MVP));
    glUniform4f(m_startColorID, 1.f, 0.f, 1.f, 1.f);
    glUniform1i(m_interpolationID, GL_FALSE);
    ExitOnGLError("ERROR! Couldn't load variables to shader.");

    // bind the vbo buffer
    glBindBuffer(GL_ARRAY_BUFFER, m_cur_context->m_focus_point_vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, // attribute 0 (must match the layout in the shader).
        3, // size
        GL_FLOAT, // type
        GL_FALSE, // normalized?
        0, // stride
        (void*) 0 // array buffer offset
        );
    glPointSize( 4.0 );
    glDrawArrays(GL_POINTS, 0, 1);
    glPointSize( 1.0 );
  }
}

void Visualizer::drawHelperPrimitives()
{
  mat4 MVP = m_cur_context->m_camera->getProjectionMatrix() * m_cur_context->m_camera->getViewMatrix();
  glUniformMatrix4fv(m_vpID, 1, GL_FALSE, glm::value_ptr(MVP));
  glUniform1i(m_interpolationID, GL_FALSE);
  ExitOnGLError("ERROR! Couldn't load variables to shader.");

  for (uint32_t i = 0; i < m_primitives.size(); i++)
  {
//    glUniform4fv(m_startColorID, 1, glm::value_ptr(primitives[i]->getColor()));
//    primitives[i]->draw();

    glPolygonOffset(-1.f, -1.f);
    glEnable(GL_POLYGON_OFFSET_LINE);
//set color for the lines.
    glm::vec4 c1 = m_primitives[i]->getColor() * 0.5f;
    glUniform4fv(m_startColorID, 1, glm::value_ptr(c1));

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    m_primitives[i]->draw(1, m_cur_context->m_lighting);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDisable(GL_POLYGON_OFFSET_LINE);
    ExitOnGLError("ERROR! Couldn't draw the edges of the primitives.");
  }

}

void Visualizer::drawPrimitivesFromSharedMem()
{
  if (m_shm_manager_primitive_arrays != NULL)
  {
    for(size_t prim_array_num = 0; prim_array_num < m_cur_context->m_prim_arrays.size(); prim_array_num++)
    {
      if(m_cur_context->m_prim_arrays[prim_array_num]->m_draw_context)
      {
        PrimitiveArrayContext* con = m_cur_context->m_prim_arrays[prim_array_num];
        // only read the buffer again if it has changed
        if(m_shm_manager_primitive_arrays->hasPrimitiveBufferChanged(prim_array_num))
        {
          glm::vec4* dev_ptr_positions;
          primitive_array::PrimitiveType tmp_prim_type = primitive_array::ePRIM_INITIAL_VALUE;
          if (m_shm_manager_primitive_arrays->getPrimitivePositions(prim_array_num, (Vector4f**)&dev_ptr_positions, con->m_total_num_voxels,
                                                              tmp_prim_type))
          {
            if(con->m_cur_vbo_size != con->m_total_num_voxels * SIZE_OF_TRANSLATION_VECTOR)
            {
              con->m_cur_vbo_size = con->m_total_num_voxels * SIZE_OF_TRANSLATION_VECTOR;
              // generate new buffer after deleting the old one.
              glDeleteBuffers(1, &(con->m_vbo));
              glGenBuffers(1, &(con->m_vbo));
              glBindBuffer(GL_ARRAY_BUFFER, con->m_vbo);
              glBufferData(GL_ARRAY_BUFFER, con->m_cur_vbo_size, 0, GL_STATIC_DRAW);
              glBindBuffer(GL_ARRAY_BUFFER, 0);
            }
            // copy the data from the list into the OpenGL buffer
            struct cudaGraphicsResource* cuda_res;
            HANDLE_CUDA_ERROR(
                cudaGraphicsGLRegisterBuffer(&cuda_res, con->m_vbo,
                                             cudaGraphicsRegisterFlagsWriteDiscard));
            glm::vec4 *vbo_ptr;
            size_t num_bytes;
            HANDLE_CUDA_ERROR(cudaGraphicsMapResources(1, &(cuda_res), 0));
            HANDLE_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer((void ** )&vbo_ptr, &num_bytes, cuda_res));

            cudaMemcpy(vbo_ptr, dev_ptr_positions, con->m_total_num_voxels * SIZE_OF_TRANSLATION_VECTOR,
                       cudaMemcpyDeviceToDevice);

            cudaIpcCloseMemHandle(dev_ptr_positions);
            HANDLE_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cuda_res, 0));
            HANDLE_CUDA_ERROR(cudaGraphicsUnregisterResource(cuda_res));
            // update the changed variable in the shared memory
            m_shm_manager_primitive_arrays->setPrimitiveBufferChangedToFalse(prim_array_num);

            // generate the corresponding default primitive type if it was not set before
            if(con->m_prim_type != tmp_prim_type)
            {
              con->m_prim_type = tmp_prim_type;
              if (con->m_prim_type == primitive_array::ePRIM_SPHERE)
              {
                Sphere* sphere;
                m_interpreter->getDefaultSphere(sphere);
                con->m_default_prim = sphere;
                con->m_default_prim->create(m_cur_context->m_lighting);
              }
              else if (con->m_prim_type == primitive_array::ePRIM_CUBOID)
              {
                Cuboid* cuboid;
                m_interpreter->getDefaultCuboid(cuboid);
                con->m_default_prim = cuboid;
                con->m_default_prim->create(m_cur_context->m_lighting);
              }
              else
              {
                LOGGING_WARNING_C(Visualization, Visualizer,
                                  "Primitive type not supported yet ... add it here." << endl);
              }
            }
          }else{
            // if it was not possible to load data from shared memory
            LOGGING_ERROR_C(Visualization, Visualizer,
                              "It was not possible to load primitive data from shared memory." << endl);
            return;
          }
        }
        if (con->m_default_prim != NULL)
        {
          GLuint color_id;
          mat4 V = m_cur_context->m_camera->getViewMatrix();
          mat4 VP = m_cur_context->m_camera->getProjectionMatrix() * V;
          if (m_cur_context->m_lighting)
          { // set up the correct variables for the shader with lighting
            glUseProgram(m_lighting_programID);
            mat4 V_inv_trans = glm::transpose(glm::inverse(V));
            vec3 lightpos_world = m_cur_context->m_camera->getCameraPosition()
                + vec3(1.f) * m_cur_context->m_camera->getCameraRight();
            vec3 light_intensity = vec3(m_cur_context->m_light_intensity);

            glUniformMatrix4fv(m_light_vpID, 1, GL_FALSE, value_ptr(VP));
            glUniformMatrix4fv(m_light_vID, 1, GL_FALSE, value_ptr(V));
            glUniformMatrix4fv(m_light_v_inv_transID, 1, GL_FALSE, value_ptr(V_inv_trans));

            glUniform3fv(m_light_lightposID, 1, glm::value_ptr(lightpos_world));
            glUniform3fv(m_light_light_intensity, 1, glm::value_ptr(light_intensity));
            glUniform1i(m_light_interpolationID, GL_FALSE);

            color_id = m_light_startColorID;
          }
          else
          {
            // set up the correct variables for the shader without lighting
            glUseProgram(m_programID);
            glUniformMatrix4fv(m_vpID, 1, GL_FALSE, value_ptr(VP));
            glUniform1i(m_interpolationID, GL_FALSE);
            color_id = m_startColorID;
          }
          ExitOnGLError("ERROR! Couldn't load variables to shader for the primitives from shared mem.");
          glBindBuffer(GL_ARRAY_BUFFER, con->m_vbo);
          glEnableVertexAttribArray(2);
          glVertexAttribDivisor(2, 1);
          glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*) 0);

          if (m_cur_context->m_draw_filled_triangles)
          {
            glUniform4fv(color_id, 1, glm::value_ptr(con->m_default_prim->getColor()));
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            con->m_default_prim->draw(con->m_total_num_voxels, m_cur_context->m_lighting);
          }
          ExitOnGLError("ERROR! Couldn't draw the filled triangles of the primitives.");
          if (m_cur_context->m_draw_edges_of_triangels)
          {
            glPolygonOffset(-1.f, -1.f);
            glEnable(GL_POLYGON_OFFSET_LINE);
            glm::vec4 c = con->m_default_prim->getColor() * 0.5f;
            glUniform4fv(color_id, 1, glm::value_ptr(c));
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            con->m_default_prim->draw(con->m_total_num_voxels, m_cur_context->m_lighting);
            glDisable(GL_POLYGON_OFFSET_LINE);
          }
          ExitOnGLError("ERROR! Couldn't draw the edges of the primitives.");

        }

      } // end of draw context
    } // end for each prim array
  } // endif (m_shm_manager_primitive_arrays != NULL)
  // reset used stuff...
  glVertexAttribDivisor(2, 0);
  glDisableVertexAttribArray(2);
  glUseProgram(0);

}

void Visualizer::drawTextLine(std::string text, int x, int y)
{
  glMatrixMode(GL_PROJECTION);
  double* matrix = new double[16];
  glGetDoublev(GL_PROJECTION_MATRIX, matrix);
  glLoadIdentity();
  glOrtho(0, getWidth(), 0, getHeight(), -5, 5);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glPushMatrix();
  glLoadIdentity();
  glRasterPos2i(x,y);
  for(uint i = 0; i < text.size(); i++)
  {
    glutBitmapCharacter(GLUT_BITMAP_8_BY_13, (int)text[i]);
  }
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glLoadMatrixd(matrix);
  glMatrixMode(GL_MODELVIEW);
}

void Visualizer::drawText(std::string text, int x, int y)
{
  std::vector<std::string> lines;
  std::string rest = text;
  size_t index = rest.find("\n");
  while(index != std::string::npos)
  {
    lines.push_back(rest.substr(0, index));
    rest = rest.substr(index + 1, rest.size() - 1);
    index = rest.find("\n");
  }
  lines.push_back(rest);
  for(uint i = 0; i < lines.size(); i++)
  {
    drawTextLine(lines[i], x, y - 15 * i);
  }
}

///////////////////////////////Begin: Callback functions for freeglut///////////////////////////////
void Visualizer::renderFunction(void)
{
  m_frameCount++;
  int32_t lastTime = glutGet(GLUT_ELAPSED_TIME);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  //draw text on display
  if (m_drawTextAll)
  {
    std::stringstream displayText;
    if(m_drawPointCountText)
    displayText << printNumberOfVoxelsDrawn() << "\n";
    if(m_drawVBOText)
    displayText << printTotalVBOsizes() << "\n";

    if (m_drawVoxelMapText)
    {
      displayText << "VoxelMaps: ";
      for (size_t i = 0; i < m_cur_context->m_voxel_maps.size(); i++)
      {
        if (!m_cur_context->m_voxel_maps[i]->m_draw_context)
        {
          displayText << "(" << m_cur_context->m_voxel_maps[i]->m_map_name << ") ";
        }
        else
        {
          displayText << m_cur_context->m_voxel_maps[i]->m_map_name << " ";
        }
      }
      displayText << "\n";
    }
    if (m_drawVoxelListText)
    {
      displayText << "VoxelLists: ";
      for (size_t i = 0; i < m_cur_context->m_voxel_lists.size(); i++)
      {
        if (!m_cur_context->m_voxel_lists[i]->m_draw_context)
        {
          displayText << "(" << m_cur_context->m_voxel_lists[i]->m_map_name << ") ";
        }
        else
        {
          displayText << m_cur_context->m_voxel_lists[i]->m_map_name << " ";
        }
      }
      displayText << "\n";
    }
    if (m_drawOctreeText)
    {
      displayText << "Octrees: ";
      for (size_t i = 0; i < m_cur_context->m_octrees.size(); i++)
      {
        if (!m_cur_context->m_octrees[i]->m_draw_context)
        {
          displayText << "(" << m_cur_context->m_octrees[i]->m_map_name << ") ";
        }
        else
        {
          displayText << m_cur_context->m_octrees[i]->m_map_name << " ";
        }
      }
      displayText << "\n";
    }
    if (m_drawPrimitiveArrayText)
    {
      displayText << "Primitive Arrays: ";
      for (size_t i = 0; i < m_cur_context->m_prim_arrays.size(); i++)
      {
        if (!m_cur_context->m_prim_arrays[i]->m_draw_context)
        {
          displayText << "(" << m_cur_context->m_prim_arrays[i]->m_map_name << ") ";
        }
        else
        {
          displayText << m_cur_context->m_prim_arrays[i]->m_map_name << " ";
        }
      }
      displayText << "\n";
    }

    if (m_drawTypeText)
    {
      displayText << "Types drawn: ";
      int count = 0;
      for (size_t i = 0; i < m_cur_context->m_draw_types.size(); i++)
      {
        if (m_cur_context->m_draw_types[i])
        {
          displayText << i << " ";
          count++;
        }
        if (count >= 35)
        {
          displayText << "\n";
          count = 0;
        }
      }
      displayText << "\n";
    }
    drawText(displayText.str(), 10, getHeight() - 20);

    if (m_drawClickedVoxelInfo)
    {
      drawText(m_clickedVoxelInfo, getWidth() / 2, 80);
    }

    if (m_drawCameraInfo)
    {
      drawText(m_cur_context->m_camera->getCameraInfo(), 10, 80);
    }
  }
  //end of drawing text

  drawPrimitivesFromSharedMem();

  glUseProgram(m_programID);

  if (m_cur_context->m_getCamTargetFromShrMem)
  {
    glm::vec3 target;
    if (m_shm_manager_visualizer != NULL && m_shm_manager_visualizer->getCameraTargetPoint(target)
        && target != m_cur_context->m_camera->getCameraTarget())
    {
      m_cur_context->m_camera->setCameraTarget(target);
      createFocusPointVBO();
    }
  }

  if (m_cur_context->m_draw_grid)
  {
    drawFocusPoint();
    drawGrid();
  }
  drawHelperPrimitives();

// set some of the unfirom shader variables for the data contexts
  mat4 VP = m_cur_context->m_camera->getProjectionMatrix() * m_cur_context->m_camera->getViewMatrix();
  if (m_cur_context->m_lighting)
  {
    glUseProgram(m_lighting_programID);
    mat4 V = m_cur_context->m_camera->getViewMatrix();
    mat4 V_inv_trans = glm::transpose(glm::inverse(V));
    glUniformMatrix4fv(m_light_vpID, 1, GL_FALSE, value_ptr(VP));
    glUniformMatrix4fv(m_light_vID, 1, GL_FALSE, value_ptr(V));
    glUniformMatrix4fv(m_light_v_inv_transID, 1, GL_FALSE, value_ptr(V_inv_trans));
    glUniform3fv(m_light_interpolationLengthID, 1, value_ptr(vec3(m_cur_context->m_interpolation_length)));
    glUniform1i(m_light_interpolationID, GL_TRUE);

    vec3 lightpos_world = m_cur_context->m_camera->getCameraPosition()
        + vec3(1.f) * m_cur_context->m_camera->getCameraRight();
    glUniform3fv(m_light_lightposID, 1, glm::value_ptr(lightpos_world));

    vec3 light_intensity = vec3(m_cur_context->m_light_intensity);
    glUniform3fv(m_light_light_intensity, 1, glm::value_ptr(light_intensity));

  }
  else
  {
    glUniformMatrix4fv(m_vpID, 1, GL_FALSE, glm::value_ptr(VP));
    glUniform3fv(m_interpolationLengthID, 1, value_ptr(vec3(m_cur_context->m_interpolation_length)));
    glUniform1i(m_interpolationID, GL_TRUE);
  }
  ExitOnGLError("ERROR! Couldn't load variables to shader.");
//////////////////////////////////////////////draw all registered voxel maps/////////////////////////////////////////
  bool set_view_to_false = true;

  // if this option is enabled, provider programs can trigger which maps should be drawn
  if (m_use_external_draw_type_triggers && (m_shm_manager_visualizer != NULL))
  {
    DrawTypes d = m_shm_manager_visualizer->getDrawTypes();
    bool changed = false;
    for(uint32_t i = eBVM_SWEPT_VOLUME_START + 1; i < MAX_DRAW_TYPES; ++i)
    {
      if (d.draw_types[i] != m_cur_context->m_draw_types[i])
      {
        m_cur_context->m_draw_types[i] = d.draw_types[i];
        changed = true;
      }
    }

    if (changed)
    {
      for (uint32_t i = 0; i < m_cur_context->m_voxel_maps.size(); i++)
        m_cur_context->m_voxel_maps[i]->m_has_draw_type_flipped = true;

      for (uint32_t i = 0; i < m_cur_context->m_octrees.size(); i++)
        m_cur_context->m_octrees[i]->m_has_draw_type_flipped = true;

      for (uint32_t i = 0; i < m_cur_context->m_voxel_lists.size(); i++)
        m_cur_context->m_voxel_lists[i]->m_has_draw_type_flipped = true;

      m_cur_context->m_camera->setViewChanged(true);
      copyDrawTypeToDevice();
    }
  }

  for (uint32_t i = 0; i < m_cur_context->m_voxel_maps.size(); i++)
  {
    if (m_cur_context->m_voxel_maps[i]->m_voxelMap != NULL && m_cur_context->m_voxel_maps[i]->m_draw_context)
    {
      if (m_cur_context->m_camera->hasViewChanged() || m_shm_manager_voxelmaps->hasVoxelMapDataChanged(i))
      {/*only update the VBO if the view has changed or new data is available*/
        /////////////////////////////////// fill up the vbo /////////////////////////////////////////////
        bool suc = fillGLBufferWithoutPrecounting(m_cur_context->m_voxel_maps[i]);
        if (suc)
        {/*only if the map was drawn completely ...*/
          m_shm_manager_voxelmaps->setVoxelMapDataChangedToFalse(i);
        }
        // only set the view to false if all maps where successfully drawn
        set_view_to_false = set_view_to_false && suc;
      }
////////////////////////////////// draw the maps //////////////////////////////////////////
      drawDataContext(m_cur_context->m_voxel_maps[i]);
    }
  }
  m_cur_context->m_camera->setViewChanged(!set_view_to_false);


////////////////////////////////draw all voxellists ///////////////////////////////////////

  for (uint32_t i = 0; i < m_cur_context->m_voxel_lists.size(); i++)
  {
    if (m_cur_context->m_voxel_lists[i]->m_draw_context)
    {
      if (m_shm_manager_voxellists->hasBufferSwapped(i))
      {
        /*Don't fill the vbo if it couldn't be updated.*/
        if (updateVoxelListContext(m_cur_context->m_voxel_lists[i], i))
        {
          fillGLBufferWithCubelist(m_cur_context->m_voxel_lists[i], i);
          m_cur_context->m_voxel_lists[i]->unmapCubesShm();
        }
        m_shm_manager_voxellists->setBufferSwappedToFalse(i);
      }
      drawDataContext(m_cur_context->m_voxel_lists[i]);
    }
  }

////////////////////////////////draw all octrees ///////////////////////////////////////

  for (uint32_t i = 0; i < m_cur_context->m_octrees.size(); i++)
  {
    if (m_cur_context->m_octrees[i]->m_draw_context)
    {
      if (m_shm_manager_octrees->hasOctreeBufferSwapped(i))
      {
        /*Don't fill the vbo if it couldn't be updated.*/
        if (updateOctreeContext(m_cur_context->m_octrees[i], i))
        {
          fillGLBufferWithCubelist(m_cur_context->m_octrees[i], i);
          cudaIpcCloseMemHandle(m_cur_context->m_octrees[i]->getCubesDevicePointer());
        }
        m_shm_manager_octrees->setOctreeBufferSwappedToFalse(i);
      }
      drawDataContext(m_cur_context->m_octrees[i]);
    }
  }

  glutSwapBuffers();
  glutPostRedisplay();
  glUseProgram(0);

// Compute time difference between current and last frame
  int32_t currentTime = glutGet(GLUT_ELAPSED_TIME);
  m_delta_time = float(currentTime - lastTime);

  if (m_max_fps != 0 && m_delta_time < 1000.f / m_max_fps)
  {
    int wait = 1000 / m_max_fps - m_delta_time;
    m_delta_time += wait;
    boost::this_thread::sleep(boost::posix_time::milliseconds(wait));
  }
}

void Visualizer::mouseClickFunction(int32_t button, int32_t state, int32_t xpos, int32_t ypos)
{
  float speed_multiplier = 80.f;
  // A left click always looks up voxel info
  if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
  {
    m_clickedVoxelInfo = printPositionOfVoxelUnderMouseCursor(xpos, ypos);
  }

  if (button == 3 && state == GLUT_DOWN) /*mouse wheel up*/
  {
    int32_t modi = glutGetModifiers();
    if (modi & GLUT_ACTIVE_CTRL  && modi & GLUT_ACTIVE_ALT)
    {
      m_cur_context->m_camera->moveAlongDirection(speed_multiplier);
    }
    else
    {
      increaseSuperVoxelSize();
    }
  }
  else if (button == 4 && state == GLUT_DOWN)/*mouse wheel down*/
  {
    int32_t modi = glutGetModifiers();
    if (modi & GLUT_ACTIVE_CTRL  && modi & GLUT_ACTIVE_ALT)
    {
      m_cur_context->m_camera->moveAlongDirection(-speed_multiplier);
    }
    else
    {
      decreaseSuperVoxelSize();
    }
  }
  else if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
  {
    int32_t modi = glutGetModifiers();
    if (modi & GLUT_ACTIVE_CTRL  && modi & GLUT_ACTIVE_ALT)
    {
      m_move_focus_enabled = true;
    }
  }
  else if (button == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN)
  {
    int32_t modi = glutGetModifiers();
    if (modi & GLUT_ACTIVE_CTRL  && modi & GLUT_ACTIVE_ALT)
    {
      m_move_focus_vertical_enabled = true;
    }
  }
  else if (button == GLUT_LEFT_BUTTON && state == GLUT_UP)
  {
    m_move_focus_enabled = false;
  }
  else if (button == GLUT_MIDDLE_BUTTON && state == GLUT_UP)
  {
    m_move_focus_vertical_enabled = false;
  }

}

/*
 * The xpos and ypos parameters indicate the mouse location in window relative coordinates.
 */
void Visualizer::mouseMotionFunction(int32_t xpos, int32_t ypos)
{
  if (m_move_focus_enabled)
  {
    m_cur_context->m_camera->moveFocusPointFromMouseInput(xpos, ypos);
    createFocusPointVBO();
  }
  else if (m_move_focus_vertical_enabled)
  {
    m_cur_context->m_camera->moveFocusPointVerticalFromMouseInput(xpos, ypos);
    createFocusPointVBO();
  }
  else
  {
    m_cur_context->m_camera->updateViewMatrixFromMouseInput(xpos, ypos);
  }
}

void Visualizer::mousePassiveMotionFunction(int32_t xpos, int32_t ypos)
{
  m_cur_context->m_camera->setMousePosition(xpos, ypos);
}

void Visualizer::resizeFunction(int32_t width, int32_t height)
{
  m_cur_context->m_camera->resizeWindow(width, height);
  glViewport(0, 0, width, height);
}
/*
 * @param callback: should be the timerFunction itself
 */
void Visualizer::timerFunction(int32_t value, void (*callback)(int32_t))
{
  if (0 != value)
  {
    std::stringstream s;
    s << m_window_title << ": " << m_frameCount * 4 << " Frames Per Second @ "
      << getWidth() << " x " << getHeight()
      << " Super Voxel Size: " << m_cur_context->m_dim_svoxel.x
      << " Keyboard Mode: " << keyboardModetoString(m_keyboardmode);
    glutSetWindowTitle(s.str().c_str());

    m_cur_fps = m_frameCount;
  }
  m_frameCount = 0;
  glutTimerFunc(250, callback, 1);
}

void Visualizer::rotate_slice_axis() {
  // update axis:
  // 0 none, 1 x, 2 y, 3 z
  m_cur_context->m_slice_axis++;
  if (m_cur_context->m_slice_axis > 3) {
    m_cur_context->m_slice_axis = 0;
  }

  // find equivalent to m_max_voxelmap_dim for octrees and voxellists?
  uint32_t max_voxelmap_dim;
  switch (m_cur_context->m_slice_axis) {
    case 1: max_voxelmap_dim = m_cur_context->m_max_voxelmap_dim.x; break;
    case 2: max_voxelmap_dim = m_cur_context->m_max_voxelmap_dim.y; break;
    case 3: max_voxelmap_dim = m_cur_context->m_max_voxelmap_dim.z; break;
    default: max_voxelmap_dim = 0; break;
  }

  // initialize axisposition to some value. guess: interesting things are near the midst of the map
  if (m_cur_context->m_slice_axis == 1 && m_cur_context->m_slice_axis_position == 0) {
    m_cur_context->m_slice_axis_position = max_voxelmap_dim / 2;
  }

  updateStartEndViewVoxelIndices(); // modify draw_whole_map case
  m_cur_context->m_camera->setViewChanged(true);
}

void Visualizer::move_slice_axis(int offset) {
  // update position
  m_cur_context->m_slice_axis_position += offset;

  if (m_cur_context->m_slice_axis_position < 0) { //prevents bugs caused by unsigned m_min_xyz_to_draw processing
    m_cur_context->m_slice_axis_position = 0;
  }

// //disabled: would limit min/max value of slice_axis_position; correct values are sometimes hard to discover
//
//  // find equivalent to m_max_voxelmap_dim for octrees and voxellists?
//  uint32_t max_voxelmap_dim =
//      (m_cur_context->m_slice_axis == 1) ?
//        (m_cur_context->m_max_voxelmap_dim.x) :
//        ((m_cur_context->m_slice_axis == 2) ?
//          (m_cur_context->m_max_voxelmap_dim.y) :
//          ((m_cur_context->m_slice_axis == 3) ?
//             (m_cur_context->m_max_voxelmap_dim.z) :
//             0));
//  if (m_cur_context->m_slice_axis_position < 0) {
//    m_cur_context->m_slice_axis_position = 0;
//  } else if (m_cur_context->m_slice_axis_position >= max_voxelmap_dim) {
//    m_cur_context->m_slice_axis_position = max_voxelmap_dim; // get a better value for max size?
//  }

  updateStartEndViewVoxelIndices();
  m_cur_context->m_camera->setViewChanged(true);
}

void Visualizer::keyboardFunction(unsigned char key, int32_t x, int32_t y)
{
  static int8_t decimal_prefix(0);
  static int8_t decimal_key(0);

  float multiplier = 1.f;
  if (glutGetModifiers() == GLUT_ACTIVE_SHIFT)
  {
    multiplier *= 5.f;
  }
  else if (glutGetModifiers() == GLUT_ACTIVE_CTRL)
  {
    multiplier *= 10.f;
  }

  int8_t modi = glutGetModifiers();
  bool alt_pressed = modi & GLUT_ACTIVE_ALT;
  printf("Keycode: %c, Modifier value: %d, alt_pressed=%u\n", key, modi, alt_pressed);

  switch (key)
  {
    case 't':
    {
      if (alt_pressed) {
        //change distance drawmode
        m_cur_context->m_distance_drawmode = (m_cur_context->m_distance_drawmode + 1) % DISTANCE_DRAW_MODE_COUNT;

        const char* mode_name = "";
        switch (m_cur_context->m_distance_drawmode) {
          case DISTANCE_DRAW_DEFAULT:             mode_name = "default drawing type 1 only"; break;
          case DISTANCE_DRAW_PBA_INTERMEDIATE:    mode_name = "pba intermediate using swept types 10-13"; break; // This is disabled. See common_defines.h
          case DISTANCE_DRAW_TWOCOLOR_GRADIENT:   mode_name = "two-color distance gradient using swept types 21-70"; break;
          case DISTANCE_DRAW_MULTICOLOR_GRADIENT: mode_name = "multi-color distance colors using swept types 11-20"; break;
          case DISTANCE_DRAW_VORONOI_LINEAR:      mode_name = "voronoi linear colors using swept types 21-70"; break;
          case DISTANCE_DRAW_VORONOI_SCRAMBLE:    mode_name = "voronoi scrambled colors using swept types 20-215"; break;
          default: mode_name = "invalid value"; break;
        }       
        std::cout << "distance_drawmode: " << mode_name << ". press 's' twice to update view." << std::endl;

      } else {
        //rotate slice axis
        rotate_slice_axis();

        const char* axis_name = "";
        switch (m_cur_context->m_slice_axis) {
          case 0: axis_name = "none"; break;
          case 1: axis_name = "x"; break;
          case 2: axis_name = "y"; break;
          case 3: axis_name = "z"; break;
          default: axis_name = "invalid value"; break;
        }
        std::cout << "slice_axis: " << axis_name  << std::endl;

        std::cout << "m_view_start_voxel_pos: " << (m_cur_context->m_view_start_voxel_pos.x) << " / " << (m_cur_context->m_view_start_voxel_pos.y) << " / " << (m_cur_context->m_view_start_voxel_pos.z);
        std::cout << ", m_view_end_voxel_pos: " << (m_cur_context->m_view_end_voxel_pos.x) << " / " << (m_cur_context->m_view_end_voxel_pos.y) << " / " << (m_cur_context->m_view_end_voxel_pos.z) << std::endl;
      }
      break;
    }
    case 'q':
      if (m_cur_context->m_slice_axis == 0) {
        std::cout << "no slice axis selected. use 't' to set a slice axis" << std::endl;
      } else {
        move_slice_axis(+1);
        std::cout << "m_cur_context->m_slice_axis_position: " << m_cur_context->m_slice_axis_position << std::endl;
        std::cout << "m_view_start_voxel_pos: " << (m_cur_context->m_view_start_voxel_pos.x) << " / " << (m_cur_context->m_view_start_voxel_pos.y) << " / " << (m_cur_context->m_view_start_voxel_pos.z);
        std::cout << ", m_view_end_voxel_pos: " << (m_cur_context->m_view_end_voxel_pos.x) << " / " << (m_cur_context->m_view_end_voxel_pos.y) << " / " << (m_cur_context->m_view_end_voxel_pos.z) << std::endl;
      }
      break;
    case 'a':
      if (m_cur_context->m_slice_axis == 0) {
        std::cout << "no slice axis selected. use 't' to set a slice axis" << std::endl;
      } else {
        move_slice_axis(-1);
        std::cout << "m_cur_context->m_slice_axis_position: " << m_cur_context->m_slice_axis_position << std::endl;
        std::cout << "m_view_start_voxel_pos: " << (m_cur_context->m_view_start_voxel_pos.x) << " / " << (m_cur_context->m_view_start_voxel_pos.y) << " / " << (m_cur_context->m_view_start_voxel_pos.z);
        std::cout << ", m_view_end_voxel_pos: " << (m_cur_context->m_view_end_voxel_pos.x) << " / " << (m_cur_context->m_view_end_voxel_pos.y) << " / " << (m_cur_context->m_view_end_voxel_pos.z) << std::endl;
      }
      break;
    case 'b':
      printNumberOfVoxelsDrawn();
      break;
    case 'c': // toggle camera mode
      m_cur_context->m_camera->toggleCameraMode();
      break;
    case 'd': //Toggle draw grid
      m_cur_context->m_draw_grid = !m_cur_context->m_draw_grid;
      break;
    case 'e': //Toggle draw triangles edges and / or filling
      keyboardDrawTriangles();
      break;
    case 'g': //Toggle draw whole map
      m_cur_context->m_draw_whole_map = !m_cur_context->m_draw_whole_map;
      if (m_cur_context->m_draw_whole_map)
        std::cout << "Drawing the whole map." << std::endl;
      else
        std::cout << "Drawing viewing range of the map." << std::endl;
      break;
    case 'h':
      printHelp();
      break;
    case 'i':
      //Toggle depth function for the collision type
      m_cur_context->m_draw_collison_depth_test_always = !m_cur_context->m_draw_collison_depth_test_always;
      if (m_cur_context->m_draw_collison_depth_test_always)
        std::cout << "Use GL_ALWAYS as the depth test function for collision type." << std::endl;
      else
        std::cout << "Use GL_LEQUAL as the depth test function for collision type." << std::endl;
      break;
    case 'k':
      m_cur_context->m_getCamTargetFromShrMem = !m_cur_context->m_getCamTargetFromShrMem;
      break;
    case 'l':
      toggleLighting();
      break;
    case 'm':
      printTotalVBOsizes();
      break;
    case 'n':
      std::cout << getDeviceMemoryInfo();
      break;
    case 'o':
      flipExternalVisibilityTrigger();
      break;
    case 'p':
      m_cur_context->m_camera->printCameraPosDirR();
      break;
    case 'r': // reset camera
      m_cur_context->m_camera->resetToInitialValues();
      createFocusPointVBO();
      break;
    case 's': // draw all swept volume types
      flipDrawSweptVolume();
      break;
    case 'v':
      printViewInfo();
      break;
    case ',':
      lastKeyboardMode();
      break;
    case '.':
      nextKeyboardMode();
      break;
    case '+': // increase light intensity
      m_cur_context->m_light_intensity += 20.f * multiplier;
      break;
    case '-': // decrease light intensity
      m_cur_context->m_light_intensity -= 20.f * multiplier;
      m_cur_context->m_light_intensity = max(m_cur_context->m_light_intensity, 0.f);
      break;
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
      if (alt_pressed)
      {
        // Set the times 10 digit
        decimal_prefix = 10 * (key - '0');
      }
      else
      {
        // Set the times 1 digit
        decimal_key = key - '0';
      }

      flipDrawType(BitVoxelMeaning(decimal_prefix + decimal_key));
      break;
  }
}

void Visualizer::keyboardSpecialFunction(int32_t key, int32_t x, int32_t y)
{
  //int8_t map_offset = 0;
  float speed_multiplier = m_cur_context->m_dim_svoxel.x * 0.3f;

  if (glutGetModifiers() == GLUT_ACTIVE_SHIFT)
  {
    speed_multiplier *= 10.f;
  }
  else if (glutGetModifiers() == GLUT_ACTIVE_CTRL)
  {
    speed_multiplier *= 30.f;
  }
  else if (glutGetModifiers() == GLUT_ACTIVE_ALT)
  {
    // add 4 to map ID
    //map_offset = 4;
  }

  switch (key)
  {
    case GLUT_KEY_UP:
      m_cur_context->m_camera->moveAlongDirection(speed_multiplier);
      break;
    case GLUT_KEY_DOWN:
      m_cur_context->m_camera->moveAlongDirection(-speed_multiplier);
      break;
    case GLUT_KEY_RIGHT:
      m_cur_context->m_camera->moveAlongRight(speed_multiplier);
      break;
    case GLUT_KEY_LEFT:
      m_cur_context->m_camera->moveAlongRight(-speed_multiplier);
      break;
    case GLUT_KEY_PAGE_UP:
      m_cur_context->m_camera->moveAlongUp(speed_multiplier);
      break;
    case GLUT_KEY_PAGE_DOWN:
      m_cur_context->m_camera->moveAlongUp(-speed_multiplier);
      break;
    case GLUT_KEY_F1:
      keyboardFlipVisibility(0);
      break;
    case GLUT_KEY_F2:
      keyboardFlipVisibility(1);
      break;
    case GLUT_KEY_F3:
      keyboardFlipVisibility(2);
      break;
    case GLUT_KEY_F4:
      keyboardFlipVisibility(3);
      break;
    case GLUT_KEY_F5:
      keyboardFlipVisibility(4);
      break;
    case GLUT_KEY_F6:
      keyboardFlipVisibility(5);
      break;
    case GLUT_KEY_F7:
      keyboardFlipVisibility(6);
      break;
    case GLUT_KEY_F8:
      keyboardFlipVisibility(7);
      break;
    case GLUT_KEY_F9:
      keyboardFlipVisibility(8);
      break;
    case GLUT_KEY_F10:
      keyboardFlipVisibility(9);
      break;
    case GLUT_KEY_F11:
      keyboardFlipVisibility(10);
      break;
    case GLUT_KEY_F12:
      keyboardFlipVisibility(11);
      break;
    default:
      break;
  }
}


void Visualizer::nextKeyboardMode()
{
  if(++m_keyboardmode > 3) m_keyboardmode = 0;
  LOGGING_INFO_C(Visualization, Visualizer, "Keyboardmode set to " << keyboardModetoString(m_keyboardmode) << endl);
}

void Visualizer::lastKeyboardMode()
{
  if(--m_keyboardmode < 0) m_keyboardmode = 3;
  LOGGING_INFO_C(Visualization, Visualizer, "Keyboardmode set to " << keyboardModetoString(m_keyboardmode) << endl);
}

void Visualizer::keyboardFlipVisibility(int8_t index)
{
  switch (m_keyboardmode)
  {
    case 0:
      flipDrawVoxelmap(index);
      break;
    case 1:
      flipDrawVoxellist(index);
      break;
    case 2:
      flipDrawOctree(index);
      break;
    case 3:
      flipDrawPrimitiveArray(index);
      break;
    default:
      break;
  }
}
std::string Visualizer::keyboardModetoString(int8_t index)
{
  switch (index)
  {
    case 0:
      return "Voxelmap";
    case 1:
      return "Voxellist";
    case 2:
      return "Octree";
    case 3:
      return "Primitivearray";
    default:
      return "";
  }
}

void Visualizer::keyboardDrawTriangles()
{
  m_trianglemode =   (m_trianglemode+1)%3;
  switch (m_trianglemode)
  {
    case 0:
      m_cur_context->m_draw_edges_of_triangels = false;
      m_cur_context->m_draw_filled_triangles = true;
      break;
    case 1:
      m_cur_context->m_draw_edges_of_triangels = true;
      m_cur_context->m_draw_filled_triangles = false;
      break;
    case 2:
      m_cur_context->m_draw_edges_of_triangels = true;
      m_cur_context->m_draw_filled_triangles = true;
      break;
    default:
      break;
  }
}

void Visualizer::menuFunction(int value)
{
  switch(value)
  {
    case MENU_NONSENSE:
      break;
    case MENU_HELP:
      printHelp();
      break;
    case MENU_TEXT_ALL:
      m_drawTextAll = !m_drawTextAll;
      break;
    case MENU_TEXT_POINTS:
      m_drawPointCountText = !m_drawPointCountText;
      break;
    case MENU_TEXT_VBO:
      m_drawVBOText = !m_drawVBOText;
      break;
    case MENU_TEXT_VOXELMAPS:
      m_drawVoxelMapText = !m_drawVoxelMapText;
      break;
    case MENU_TEXT_VOXELLISTS:
      m_drawVoxelListText = !m_drawVoxelListText;
      break;
    case MENU_TEXT_OCTREES:
      m_drawOctreeText = !m_drawOctreeText;
      break;
    case MENU_TEXT_PRIMITIVEARRAYS:
      m_drawPrimitiveArrayText = !m_drawPrimitiveArrayText;
      break;
    case MENU_TEXT_TYPES:
      m_drawTypeText = !m_drawTypeText;
      break;
    case MENU_TEXT_CLICKEDVOXELINFO:
      m_drawClickedVoxelInfo = !m_drawClickedVoxelInfo;
      break;
    case MENU_CAMERA_RESET:
      m_cur_context->m_camera->resetToInitialValues();
      createFocusPointVBO();
      break;
    case MENU_CAMERA_FREE:
      m_cur_context->m_camera->m_camera_orbit = true;
      m_cur_context->m_camera->toggleCameraMode();
      break;
    case MENU_CAMERA_ORBIT:
      m_cur_context->m_camera->m_camera_orbit = false;
      m_cur_context->m_camera->toggleCameraMode();
      break;
    case MENU_CAMERA_TOGGLETEXT:
      m_drawCameraInfo = !m_drawCameraInfo;
      break;
    case MENU_GRID_ON:
      m_cur_context->m_draw_grid = true;
      break;
    case MENU_GRID_OFF:
      m_cur_context->m_draw_grid = false;
      break;
    case MENU_RENDERMODE_SOLID:
      m_cur_context->m_draw_edges_of_triangels = false;
      m_cur_context->m_draw_filled_triangles = true;
      break;
    case MENU_RENDERMODE_WIREFRAME:
      m_cur_context->m_draw_edges_of_triangels = true;
      m_cur_context->m_draw_filled_triangles = false;
      break;
    case MENU_RENDERMODE_SOLIDWIREFRAME:
      m_cur_context->m_draw_edges_of_triangels = true;
      m_cur_context->m_draw_filled_triangles = true;
      break;    
    case MENU_RENDERMODE_DIST_DEFAULT:
      m_cur_context->m_distance_drawmode = DISTANCE_DRAW_DEFAULT;
      break;
    case MENU_RENDERMODE_DIST_TWOCOLOR_GRADIENT:
      m_cur_context->m_distance_drawmode = DISTANCE_DRAW_TWOCOLOR_GRADIENT;
      break;
    case MENU_RENDERMODE_DIST_MULTICOLOR_GRADIENT:
      m_cur_context->m_distance_drawmode = DISTANCE_DRAW_MULTICOLOR_GRADIENT;
      break;
    case MENU_RENDERMODE_DIST_VORONOI_LINEAR:
      m_cur_context->m_distance_drawmode = DISTANCE_DRAW_VORONOI_LINEAR;
      break;
    case MENU_RENDERMODE_DIST_VORONOI_SCRAMBLE:
      m_cur_context->m_distance_drawmode = DISTANCE_DRAW_VORONOI_SCRAMBLE;
      break;
    case MENU_RENDERMODE_SLICING_OFF:
      m_cur_context->m_slice_axis = 0;
      break;
    case MENU_RENDERMODE_SLICING_X:
      m_cur_context->m_slice_axis = 1;
      break;
    case MENU_RENDERMODE_SLICING_Y:
      m_cur_context->m_slice_axis = 2;
      break;
    case MENU_RENDERMODE_SLICING_Z:
      m_cur_context->m_slice_axis = 3;
      break;
    case MENU_DRAWMAP_ALL:
      m_cur_context->m_draw_whole_map = true;
      break;
    case MENU_DRAWMAP_VIEW:
      m_cur_context->m_draw_whole_map = false;
      break;
    case MENU_DEPTHTEST_ALWAYS:
      m_cur_context->m_draw_collison_depth_test_always = true;
      break;
    case MENU_DEPTHTEST_LEQUAL:
      m_cur_context->m_draw_collison_depth_test_always = false;
      break;
    case MENU_LIGHT_ON:
      m_cur_context->m_lighting = false;
      break;
    case MENU_LIGHT_OFF:
      m_cur_context->m_lighting = true;
      break;
    case MENU_VISIBILITYTRIGGER_ACTIVATED:
      m_use_external_draw_type_triggers = true;
      break;
    case MENU_VISIBILITYTRIGGER_DEACTIVATED:
      m_use_external_draw_type_triggers = false;
      break;
    default:
      //Functions for maps: VoxelMaps
      if(value >= 300 && value < 400)
      {
        flipDrawVoxelmap(value - 300);
      }
      //Functions for maps: voxellists
      if(value >= 400 && value < 500)
      {
        flipDrawVoxellist(value - 400);
      }
      //Functions for maps: Octrees
      if(value >= 500 && value < 600)
      {
        flipDrawOctree(value - 500);
      }
      if(value >= 600 && value < 700)
      {
        flipDrawPrimitiveArray(value - 600);
      }
      break;
  }
  glutPostRedisplay();
  return;
}

void Visualizer::idleFunction(void) const
{
  glutPostRedisplay();
}

void Visualizer::cleanupFunction(void)
{

  for (uint32_t i = 0; i < m_cur_context->m_voxel_maps.size(); i++)
  {
    HANDLE_CUDA_ERROR(cudaGraphicsUnregisterResource(m_cur_context->m_voxel_maps[i]->m_cuda_ressources));
    deleteGLBuffer(m_cur_context->m_voxel_maps[i]);
  }
  for (uint32_t i = 0; i < m_cur_context->m_octrees.size(); i++)
  {
    HANDLE_CUDA_ERROR(cudaGraphicsUnregisterResource(m_cur_context->m_octrees[i]->m_cuda_ressources));
    deleteGLBuffer(m_cur_context->m_octrees[i]);
  }

}
///////////////////////////////End: Callback functions for freeglut///////////////////////////////
void Visualizer::increaseSuperVoxelSize()
{
  Vector3ui d = m_cur_context->m_dim_svoxel;

  d = d + d;
  d = minVec(d,Vector3ui(64));
//  LOGGING_DEBUG_C(Visualization, VoxelMapVisualizer_gpu,
//                 "dimension of super voxel: (" << d.x << ", " << d.y << ", " << d.z << ")" << endl);
  m_cur_context->m_dim_svoxel = d;
  if (m_shm_manager_octrees != NULL)
  {
    m_shm_manager_octrees->setSuperVoxelSize(d.x);
  }

  for (uint32_t i = 0; i < m_cur_context->m_voxel_maps.size(); i++)
  {
    m_cur_context->m_voxel_maps[i]->updateCudaLaunchVariables(d);
  }
  /*to force an update of VBO*/
  m_cur_context->m_camera->setViewChanged(true);
}

void Visualizer::decreaseSuperVoxelSize()
{
  Vector3ui d = m_cur_context->m_dim_svoxel;
  if (d.x > 1 && d.y > 1 && d.z > 1)
  {
    d = d / Vector3ui(2);
//    LOGGING_DEBUG_C(Visualization, VoxelMapVisualizer_gpu,
//                   "dimension of super voxel: (" << d.x << ", " << d.y << ", " << d.z << ")" << endl);
    m_cur_context->m_dim_svoxel = d;
    if (m_shm_manager_octrees != NULL)
    {
      m_shm_manager_octrees->setSuperVoxelSize(d.x);
    }
    for (uint32_t i = 0; i < m_cur_context->m_voxel_maps.size(); i++)
    {
      m_cur_context->m_voxel_maps[i]->updateCudaLaunchVariables(d);
    }
  }
  /*to force an update of VBO*/
  m_cur_context->m_camera->setViewChanged(true);
}

void Visualizer::flipDrawVoxelmap(uint32_t index)
{
  if (index >= m_cur_context->m_voxel_maps.size())
  {
    LOGGING_INFO_C(Visualization, Visualizer, "No Voxelmap registered at index " << index << endl);
    return;
  }
  VoxelmapContext* con = m_cur_context->m_voxel_maps[index];
  con->m_draw_context = !con->m_draw_context;

  if (con->m_draw_context)
  {
    m_cur_context->m_camera->setViewChanged(true); // force an update of vbos
    LOGGING_INFO_C(Visualization, Visualizer,
                   "Activated the drawing of the Voxelmap: " << con->m_map_name << endl);
  }
  else
  {
    LOGGING_INFO_C(Visualization, Visualizer,
                   "Deactivated the drawing of the Voxelmap: " << con->m_map_name << endl);
  }
}
void Visualizer::flipDrawOctree(uint32_t index)
{
  if (index >= m_cur_context->m_octrees.size())
  {
    LOGGING_INFO_C(Visualization, Visualizer, "No Octree registered at index " << index << endl);
    return;
  }
  CubelistContext* con = m_cur_context->m_octrees[index];
  con->m_draw_context = !con->m_draw_context;
  if (con->m_draw_context)
  {
    LOGGING_INFO_C(Visualization, Visualizer,
                   "Activated the drawing of the Octree: " << con->m_map_name << endl);
  }
  else
  {
    LOGGING_INFO_C(Visualization, Visualizer,
                   "Deactivated the drawing of the Octree: " << con->m_map_name << endl);
  }
}

void Visualizer::flipDrawVoxellist(uint32_t index)
{
  if (index >= m_cur_context->m_voxel_lists.size())
  {
    LOGGING_INFO_C(Visualization, Visualizer, "No Voxellist registered at index " << index << endl);
    return;
  }
  CubelistContext* con = m_cur_context->m_voxel_lists[index];
  con->m_draw_context = !con->m_draw_context;
  if (con->m_draw_context)
  {
    LOGGING_INFO_C(Visualization, Visualizer,
                   "Activated the drawing of the Voxellist: " << con->m_map_name << endl);
  }
  else
  {
    LOGGING_INFO_C(Visualization, Visualizer,
                   "Deactivated the drawing of the Voxellist: " << con->m_map_name << endl);
  }
}

void Visualizer::flipDrawPrimitiveArray(uint32_t index)
{
  if(index >= m_cur_context->m_prim_arrays.size())
  {
    LOGGING_INFO_C(Visualization, Visualizer, "No PrimitiveArray registered at index " << index << endl);
    return;
  }
  PrimitiveArrayContext* con = m_cur_context->m_prim_arrays[index];
  con->m_draw_context = !con->m_draw_context;
  if (con->m_draw_context)
  {
    LOGGING_INFO_C(Visualization, Visualizer,
                   "Activated the drawing of the PrimitiveArray: " << con->m_map_name << endl);
  }
  else
  {
    LOGGING_INFO_C(Visualization, Visualizer,
                   "Deactivated the drawing of the PrimitiveArray: " << con->m_map_name << endl);
  }
}

void Visualizer::flipExternalVisibilityTrigger()
{
  if(m_use_external_draw_type_triggers)
  {
    m_use_external_draw_type_triggers = false;
    LOGGING_INFO_C(Visualization, Visualizer, "External visibility trigger of Swept Volume subtypes deactivated" << endl);
  }else{
    m_use_external_draw_type_triggers = true;
    if (m_draw_swept_volumes)
    {
      flipDrawSweptVolume(); // They will not be drawn after deactivation, anyway
    }
    LOGGING_INFO_C(Visualization, Visualizer, "External visibility trigger of Swept Volume subtypes activated" << endl);
  }
}

void Visualizer::flipDrawType(BitVoxelMeaning type)
{
  if (m_cur_context->m_draw_types[type])
  {
    LOGGING_INFO_C(Visualization, Visualizer, "Draw type " << typeToString(type) << " deactivated" << endl);

    m_cur_context->m_draw_types[type] = 0;
  }
  else
  {
    LOGGING_INFO_C(Visualization, Visualizer, "Draw type " << typeToString(type) << " activated" << endl);
    m_cur_context->m_draw_types[type] = 1;
  }
  for (uint32_t i = 0; i < m_cur_context->m_voxel_maps.size(); i++)
  {
    m_cur_context->m_voxel_maps[i]->m_has_draw_type_flipped = true;
  }
  for (uint32_t i = 0; i < m_cur_context->m_octrees.size(); i++)
  {
    m_cur_context->m_octrees[i]->m_has_draw_type_flipped = true;
  }
  for (uint32_t i = 0; i < m_cur_context->m_voxel_lists.size(); i++)
  {
    m_cur_context->m_voxel_lists[i]->m_has_draw_type_flipped = true;
  }
  m_cur_context->m_camera->setViewChanged(true);
  copyDrawTypeToDevice();
}

void Visualizer::flipDrawSweptVolume()
{
  const uint8 start = static_cast<uint>(eBVM_SWEPT_VOLUME_START);
  const uint8 end = static_cast<uint>(eBVM_SWEPT_VOLUME_END);
  if (m_draw_swept_volumes)
  {
    m_draw_swept_volumes = false;
    LOGGING_INFO_C(Visualization, Visualizer, "Drawing complete Swept Volumes" << " deactivated" << endl);

    thrust::fill(m_cur_context->m_draw_types.begin()+start, m_cur_context->m_draw_types.begin()+end, 0);
  }
  else
  {
    m_draw_swept_volumes = true;
    LOGGING_INFO_C(Visualization, Visualizer, "Drawing complete Swept Volumes" << " activated" << endl);
    // for all from start to end
    thrust::fill(m_cur_context->m_draw_types.begin()+start, m_cur_context->m_draw_types.begin()+end, 1);
  }
  for (uint32_t i = 0; i < m_cur_context->m_voxel_maps.size(); i++)
  {
    m_cur_context->m_voxel_maps[i]->m_has_draw_type_flipped = true;
  }
  for (uint32_t i = 0; i < m_cur_context->m_octrees.size(); i++)
  {
    m_cur_context->m_octrees[i]->m_has_draw_type_flipped = true;
  }
  for (uint32_t i = 0; i < m_cur_context->m_voxel_lists.size(); i++)
  {
    m_cur_context->m_voxel_lists[i]->m_has_draw_type_flipped = true;
  }

  m_cur_context->m_camera->setViewChanged(true);
  copyDrawTypeToDevice();
}

void Visualizer::copyDrawTypeToDevice()
{
  m_cur_context->m_d_draw_types = m_cur_context->m_draw_types;
  m_cur_context->m_prefixes.resize(m_cur_context->m_draw_types.size());
  thrust::exclusive_scan(m_cur_context->m_draw_types.begin(), m_cur_context->m_draw_types.end(),
                         m_cur_context->m_prefixes.begin());
  m_cur_context->m_d_prefixes = m_cur_context->m_prefixes;

}

void Visualizer::updateTypesSegmentMapping(DataContext* context)
{
  if (context->m_has_draw_type_flipped)
  {
    for (uint32_t i = 0; i < context->m_types_segment_mapping.size(); ++i)
    {
      uint8_t type = m_cur_context->m_prefixes.size() - 1;
      while (true)
      {
        if (m_cur_context->m_draw_types[type] == 1 && m_cur_context->m_prefixes[type] == i)
        {
          context->m_types_segment_mapping[i] = (uint8_t) type;
          break;
        }
        if (type == 0) /*exit loop if type == 0 to prevent endless loop */
          break;
        type--;
      }
    }
    context->m_has_draw_type_flipped = false;
  }
}

/**
 * Prints the position of the voxel under the mouse cursor.
 */
std::string Visualizer::printPositionOfVoxelUnderMouseCursor(int32_t xpos, int32_t ypos)
{
//clear background color
  glClearColor(1.f, 1.f, 1.f, 1.f);
// if x or y or z is equal to 2^24 - 1 the clicked point should be the background
  uint32_t background_pos = pow(2.f, 24.f) - 1.f;
  Vector3ui d = m_cur_context->m_dim_svoxel;
  glm::vec3 c_pos = m_cur_context->m_camera->getCameraPosition();
  bool found_in_voxelmap = false;
  bool found_in_octree = false;
  bool found_in_voxellist = false;
  Vector3ui n_pos;
  uint32_t data_index = 0;
  float distance = FLT_MAX;

  for (uint32_t i = 0; i < m_cur_context->m_voxel_maps.size(); i++)
  {
    VoxelmapContext * con = m_cur_context->m_voxel_maps[i];
    if (!con->m_draw_context)
    {/*If this voxel map isn't drawn => don't look there for voxels*/
      continue;
    }
    Vector3ui pos = Vector3ui(getDataPositionFromColorMap(xpos, ypos, 0, con),
                              getDataPositionFromColorMap(xpos, ypos, 1, con),
                              getDataPositionFromColorMap(xpos, ypos, 2, con));
    if (pos.x == background_pos || pos.y == background_pos || pos.z == background_pos)
    {
      /*If no voxel was found in this map try the next one*/
      continue;
    }
    found_in_voxelmap = true;
    /*If a voxel was found check if it is nearer than a previously found one (relative to the camera position).*/
    glm::vec3 v_pos = convertFromVector3uiToVec3(pos);
    float distance_camera_voxel = glm::distance(c_pos, v_pos);
    if (distance_camera_voxel < distance)
    {
      distance = distance_camera_voxel;
      n_pos = pos;
      data_index = i;
    }
  }
  for (uint32_t i = 0; i < m_cur_context->m_octrees.size(); i++)
  {
    CubelistContext * con = m_cur_context->m_octrees[i];
    if (!con->m_draw_context)
    {/*If this octree isn't drawn => don't look there for voxels*/
      continue;
    }
    Vector3ui pos = Vector3ui(getDataPositionFromColorMap(xpos, ypos, 0, con),
                              getDataPositionFromColorMap(xpos, ypos, 1, con),
                              getDataPositionFromColorMap(xpos, ypos, 2, con));
    if (pos.x == background_pos || pos.y == background_pos || pos.z == background_pos)
    {
      /*If no voxel was found in this map try the next one*/
      continue;
    }
    found_in_octree = true;
    /*If a voxel was found check if it is nearer than a previously found one (relative to the camera position).*/
    glm::vec3 v_pos = convertFromVector3uiToVec3(pos);
    float distance_camera_voxel = glm::distance(c_pos, v_pos);
    if (distance_camera_voxel < distance)
    {
      distance = distance_camera_voxel;
      n_pos = pos;
      data_index = i;
      found_in_voxelmap = false;
    }
  }
  for (uint32_t i = 0; i < m_cur_context->m_voxel_lists.size(); i++)
  {
    CubelistContext * con = m_cur_context->m_voxel_lists[i];
    if (!con->m_draw_context)
    {/*If this voxellist isn't drawn => don't look there for voxels*/
      continue;
    }
    Vector3ui pos = Vector3ui(getDataPositionFromColorMap(xpos, ypos, 0, con),
                              getDataPositionFromColorMap(xpos, ypos, 1, con),
                              getDataPositionFromColorMap(xpos, ypos, 2, con));
    if (pos.x == background_pos || pos.y == background_pos || pos.z == background_pos)
    {
      /*If no voxel was found in this map try the next one*/
      continue;
    }
    found_in_voxellist = true;
    /*If a voxel was found check if it is nearer than a previously found one (relative to the camera position).*/
    glm::vec3 v_pos = convertFromVector3uiToVec3(pos);
    float distance_camera_voxel = glm::distance(c_pos, v_pos);
    if (distance_camera_voxel < distance)
    {
      distance = distance_camera_voxel;
      n_pos = pos;
      data_index = i;
      found_in_octree = false;
      found_in_voxelmap = false;
    }
  }

std::stringstream returnString;
//reset background color
  glm::vec4 bc = m_cur_context->m_background_color;
  glClearColor(bc.x, bc.y, bc.z, 1.f);
  std::cout << std::endl;
  if (!found_in_voxelmap && !found_in_octree && !found_in_voxellist)
  {
    returnString << "There is no voxel!" << std::endl;
  }
  float scale = m_cur_context->m_scale_unit.first;
  std::string unit = m_cur_context->m_scale_unit.second;
  if (found_in_voxelmap)
  {
    returnString << "Found voxel in voxel map: " << m_cur_context->m_voxel_maps[data_index]->m_map_name
        << std::endl;
    n_pos = n_pos / d;

    if (d.x > 1 || d.y > 1 || d.z > 1)
    {
// if super voxel dim > 0, print the start end end voxel
      n_pos = n_pos * d;
      Vector3ui end_pos = n_pos + d - Vector3ui(1);

      returnString << "Start voxel position x: " << n_pos.x << " y: " << n_pos.y << " z: " << n_pos.z
          << std::endl;
      returnString << "Start voxel distance x: " << n_pos.x * scale << unit << " y: " << n_pos.y * scale << unit
          << " z: " << n_pos.z * scale << unit << std::endl;
      returnString << "End voxel position x: " << end_pos.x << " y: " << end_pos.y << " z: " << end_pos.z
          << std::endl;
      returnString << "End voxel distance x: " << end_pos.x * scale << unit << " y: " << end_pos.y * scale
          << unit << " z: " << end_pos.z * scale << unit << std::endl;
    }
    else
    {
      returnString << "Voxel position x: " << n_pos.x << " y: " << n_pos.y << " z: " << n_pos.z << std::endl;

      returnString << "Voxel distance x: " << n_pos.x * scale << unit << " y: " << n_pos.y * scale << unit
          << " z: " << n_pos.z * scale << unit << std::endl;
    }

    VoxelmapContext* vm_context = m_cur_context->m_voxel_maps[data_index];
    if (vm_context->m_voxelMap->getMapType() == MT_BITVECTOR_VOXELMAP) {
      gpu_voxels::voxelmap::BitVectorVoxelMap* vm = static_cast<gpu_voxels::voxelmap::BitVectorVoxelMap *>(vm_context->m_voxelMap);

      gpu_voxels::voxelmap::BitVectorVoxelMap::Voxel voxel;
      cudaMemcpy(&voxel, gpu_voxels::voxelmap::getVoxelPtr(vm->getDeviceDataPtr(), vm->getDimensions(), n_pos), sizeof(gpu_voxels::voxelmap::BitVectorVoxelMap::Voxel), cudaMemcpyDeviceToHost);
      returnString << "Voxel info: Bitvektor = " << voxel << std::endl;

    } else if (vm_context->m_voxelMap->getMapType() == MT_PROBAB_VOXELMAP) {
      gpu_voxels::voxelmap::ProbVoxelMap* vm = static_cast<gpu_voxels::voxelmap::ProbVoxelMap *>(vm_context->m_voxelMap);

      gpu_voxels::voxelmap::ProbVoxelMap::Voxel voxel;
      cudaMemcpy(&voxel, gpu_voxels::voxelmap::getVoxelPtr(vm->getDeviceDataPtr(), vm->getDimensions(), n_pos), sizeof(gpu_voxels::voxelmap::ProbVoxelMap::Voxel), cudaMemcpyDeviceToHost);
      returnString << "Voxel info: " << "Occupancy = " << voxel << " (Probability = " << ProbabilisticVoxel::probabilityToFloat(voxel.getOccupancy()) << ")" << std::endl;
    } else if (vm_context->m_voxelMap->getMapType() == MT_DISTANCE_VOXELMAP) {
      gpu_voxels::voxelmap::DistanceVoxelMap* dvm = static_cast<gpu_voxels::voxelmap::DistanceVoxelMap *>(vm_context->m_voxelMap);

      gpu_voxels::voxelmap::DistanceVoxelMap::Voxel voxel;
      cudaMemcpy(&voxel, gpu_voxels::voxelmap::getVoxelPtr(dvm->getDeviceDataPtr(), dvm->getDimensions(), n_pos), sizeof(gpu_voxels::voxelmap::DistanceVoxelMap::Voxel), cudaMemcpyDeviceToHost);
      returnString << "Voxel info: Closest obstacle: " << voxel << std::endl;
    }
  }
  if (found_in_octree)
  {
    returnString << "Found voxel in octree: " << m_cur_context->m_octrees[data_index]->m_map_name << std::endl;
    returnString << "Voxel position x: " << n_pos.x << " y: " << n_pos.y << " z: " << n_pos.z << std::endl;
    returnString << "Voxel distance x: " << n_pos.x * scale << unit << " y: " << n_pos.y * scale << unit
        << " z: " << n_pos.z * scale << unit << std::endl;
  }
  if (found_in_voxellist)
  {
    returnString << "Found voxel in Voxellist: " << m_cur_context->m_voxel_lists[data_index]->m_map_name << std::endl;
    returnString << "Voxel position x: " << n_pos.x << " y: " << n_pos.y << " z: " << n_pos.z << std::endl;
    returnString << "Voxel distance x: " << n_pos.x * scale << unit << " y: " << n_pos.y * scale << unit
        << " z: " << n_pos.z * scale << unit << std::endl;

    // First we have to map in the CUDA shared memory of the data structure
    CubelistContext* vm_context = m_cur_context->m_voxel_lists[data_index];
    if (updateVoxelListContext(vm_context, data_index))
    {

      // as we can not access cubes by their XYZ coords, we have to search for the right voxel:
      Cube h_found_cube;
      Cube* d_found_cube;
      cudaMalloc((void**)&d_found_cube, sizeof(Cube));
      bool h_found_flag(false);
      bool* d_found_flag;
      cudaMalloc((void**)&d_found_flag, sizeof(bool));
      cudaMemset(d_found_flag, 0, sizeof(bool));

      find_cubes_by_coordinates<<<vm_context->m_num_blocks, vm_context->m_threads_per_block>>>(vm_context->getCubesDevicePointer(),
                                                                                               vm_context->getNumberOfCubes(),
                                                                                               n_pos, d_found_cube, d_found_flag);
      CHECK_CUDA_ERROR();


      HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
      HANDLE_CUDA_ERROR(cudaMemcpy((void*)&h_found_flag, d_found_flag, sizeof(bool), cudaMemcpyDeviceToHost));

      if(h_found_flag)
      {
        cudaMemcpy((void*)&h_found_cube, d_found_cube, sizeof(Cube), cudaMemcpyDeviceToHost);
        returnString << "Voxel info: " << h_found_cube.m_type_vector << std::endl;
      }

      cudaFree(d_found_cube);
      cudaFree(d_found_flag);

      // unmap the CUDA shared mem
      vm_context->unmapCubesShm();
    }
  }

  std::cout << returnString.str();
  return returnString.str();
}

void Visualizer::toggleLighting()
{
  m_cur_context->m_lighting = !m_cur_context->m_lighting;
}

__inline__ uint8_t Visualizer::typeToColorIndex(uint8_t type)
{
  /*Undefined(255) -> 255  (since it will normally not be used for visualization)*/
  /*Dynamic(254) -> 0 */
  /*Static(253)  -> 1 */
  /* .... */
  /*eBVM_SWEPT_VOLUME_START(0) -> 254 */
  //return type == (uint8_t) 255 ? (uint8_t) 255 : (uint8_t) 254 - type;
  return type;
}
/**
 * Distributes the maximum usable memory on all data structures
 */
void Visualizer::distributeMaxMemory()
{
  // as a quick hack, the maps will get 4 times the memory than the primitive arrays
  uint32_t num_data_structures = (4 * m_cur_context->m_voxel_maps.size()) +
                                 (4 * m_cur_context->m_octrees.size()) +
                                 (1 * m_cur_context->m_prim_arrays.size());
  if (num_data_structures == 0)
  {
    return;
  }
  size_t mem_per_data_structure = m_max_mem / num_data_structures;

  for (uint32_t i = 0; i < m_cur_context->m_voxel_maps.size(); i++)
  {
    m_cur_context->m_voxel_maps[i]->m_max_vbo_size = 4 * mem_per_data_structure;
  }

  for (uint32_t i = 0; i < m_cur_context->m_octrees.size(); i++)
  {
    m_cur_context->m_octrees[i]->m_max_vbo_size = 4 * mem_per_data_structure;
  }

  for (uint32_t i = 0; i < m_cur_context->m_prim_arrays.size(); i++)
  {
    m_cur_context->m_prim_arrays[i]->m_max_vbo_size = 1 * mem_per_data_structure;
  }

}

std::string Visualizer::printNumberOfVoxelsDrawn()
{
  uint32_t num_voxels = 0;

  for (uint32_t i = 0; i < m_cur_context->m_voxel_maps.size(); i++)
  {
    num_voxels += m_cur_context->m_voxel_maps[i]->m_total_num_voxels;
  }
  for (uint32_t i = 0; i < m_cur_context->m_octrees.size(); i++)
  {
    num_voxels += m_cur_context->m_octrees[i]->m_total_num_voxels;
  }
  std::stringstream returnString;
  returnString << "#Voxels: " << num_voxels << " #Triangles: " << num_voxels*36;
  //std::cout << returnString.str() << std::endl;
  return returnString.str();
}

std::string Visualizer::printTotalVBOsizes()
{
  size_t total_size = 0;
  for (size_t i = 0; i < m_cur_context->m_voxel_maps.size(); i++)
  {
    total_size += m_cur_context->m_voxel_maps[i]->m_cur_vbo_size;
  }
  for (uint32_t i = 0; i < m_cur_context->m_octrees.size(); i++)
  {
    total_size += m_cur_context->m_octrees[i]->m_cur_vbo_size;
  }
  for (uint32_t i = 0; i < m_cur_context->m_voxel_lists.size(); i++)
  {
    total_size += m_cur_context->m_voxel_lists[i]->m_cur_vbo_size;
  }
  std::stringstream returnString;
  returnString << "VBO Size: " << total_size / 1e+006 << " MByte";
  //std::cout << "Current Size of the VBOs: " << total_size / 1e+006 << " MByte.\n";
  return returnString.str();
  
}

std::string Visualizer::printViewInfo()
{
  //std::cout << "--------printing view info--------" << std::endl;
  std::stringstream resultString;
  glm::vec3 vd = m_cur_context->m_dim_view;
  resultString << "dimension of view x: " << vd.x << " y: " << vd.y << " z: " << vd.z << std::endl;

  Vector3ui vs = m_cur_context->m_view_start_voxel_pos;
  resultString << "m_view_start_voxel_index x: " << vs.x << " y: " << vs.y << " z: " << vs.z << std::endl;
  Vector3ui ve = m_cur_context->m_view_end_voxel_pos;
  resultString << "m_view_end_voxel_index x: " << ve.x << " y: " << ve.y << " z: " << ve.z << std::endl;

  glm::vec3 d = m_cur_context->m_camera->getCameraDirection();
  resultString << "Camera direction  x: " << d.x << " y: " << d.y << " z: " << d.z << std::endl;

  glm::vec3 t = m_cur_context->m_camera->getCameraTarget();
  resultString << "Camera target: x: " << t.x << " y: " << t.y << " z: " << t.z;
  return resultString.str();
}

/**
 * Prints the key bindings.
 *  * <li>h: Prints help.</li>
  * <li>a: move slice axis negative</li>
  * <li>b: print number of Voxels drawn.</li>
  * <li>c: Toggles between orbit and free-flight camera.</li>
  * <li>d: Toggles drawing of the grid.</li>
  * <li>e: Toggle through drawing modes for triangles </li>
  * <li>g: Toggles draw whole map. This disables the clipping of the field of view.</li>
  * <li>i: Toggles the OpenGL depth function for the collision type to draw colliding Voxles over all other Voxels. GL_ALWAYS should be used to get an overview (produces artifacts).</li>
  * <li>k: Toggles use of camera target point from shared memory (Currently not implemented).</li>
  * <li>l: Toggles lighting on/off.</li>
  * <li>m: Prints total VBO size in GPU memory.</li>
  * <li>n: Prints device memory info.</li>
  * <li>o: Overwrite providers possibility to trigger visibility of swept volumes: The provider may select, which Swept-Volumes are visible. This option overwrites the behaviour.</li>
  * <li>p: Print camera position. Output can directly be pasted into an XML Configfile.</li>
  * <li>q: move slice axis positive</li>
  * <li>r: Reset camera to default position.</li>
  * <li>s: Draw all swept volume types on/off (All SweptVol types will be deactivated after switching off.)</li>
  * <li>t: rotate slice axis</li>
  * <li>v: Prints view info.</li>
  * <li>+/-: Increase/decrease the light intensity. Can be multiplied with SHIFT / CTRL.</li>
  * <li>CRTL: Hold down for high movement speed.</li>
  * <li>SHIFT: Hold down for medium movement speed.</li>
  * <li>0-9: Toggles the drawing of the different Voxel-types.</li>
  * <li>,/.: previous/next keyboard mode: Voxelmap > Voxellist > Octree > Primitivearrays >
  * <li>F1-F11 Toggle drawing according to the keyboard mode.</li>
 */
void Visualizer::printHelp()
{
  std::cout << "" << std::endl;
  std::cout << "" << std::endl;
  std::cout << "Visualizer Controls" << std::endl;
  std::cout << "" << std::endl;
  std::cout << "---->Keyboard" << std::endl;
  std::cout << "h: print this help." << std::endl;
  std::cout << "b: print number of Voxels drawn." << std::endl;

  std::cout << "v: print view info." << std::endl;
  std::cout << "o: overwrite providers possibility to trigger visibility of swept volumes on/off" << std::endl;
  std::cout << "s: draw all swept volume types on/off (All SweptVol types will be deactivated after switching off.)" << std::endl;
  std::cout << "p: print camera position." << std::endl;
  std::cout << "m: print total VBO size." << std::endl;
  std::cout << "n: print device memory info." << std::endl;
  std::cout << "" << std::endl;
  std::cout << "g: toggle draw whole map." << std::endl;
  std::cout << "t: change slice axis (none/x/y/z). requires 'draw whole map mode' to show slices." << std::endl;
  std::cout << "q|a: increment/decrement slice axis position (see 't'')" << std::endl;
  std::cout << "ALT + t: change distance draw mode (default / color gradient / voronoi)." << std::endl;
  std::cout << "d: toggle draw grid." << std::endl;
  std::cout << "e: e: Toggle through drawing modes for triangles." << std::endl;
  std::cout << "l: toggle lighting on/off." << std::endl;
  std::cout
      << "i: toggle the OpenGL depth function for the collision type. GL_ALWAYS should be used to get an overview (produces artifacts)."
      << std::endl;
  std::cout << "c: toggle between orbit and free-flight camera." << std::endl;
  //std::cout << "k: toggle use of camera target point from shared memory." << std::endl;
  std::cout << "+/-: increase/decrease the light intensity. " << std::endl;
  std::cout << "r: reset camera to default position." << std::endl;
  std::cout << "CRTL: high movement speed." << std::endl;
  std::cout << "SHIFT: medium movement speed." << std::endl;
  std::cout << "0-9: toggle the drawing of the type with the value of this digit" << std::endl;
  std::cout << "ALT + 0-9: toggle the drawing of the type with 10*x + previous non-ALT-digit." << std::endl;
  std::cout << ",/.: previous/next keyboard mode: Voxelmap > Voxellist > Octree > Primitivearrays >" << std::endl;
  std::cout << "F1-F12 Toggle drawing according to the keyboard mode." << std::endl;
  std::cout << "" << std::endl;
  std::cout << "---->Mouse" << std::endl;
  std::cout << "RIGHT_BUTTON: print x,y,z coordinates of the clicked voxel." << std::endl;
  std::cout << "LEFT_BUTTON: enables mouse movement." << std::endl;
  std::cout << "ALT + CTRL + LEFT_BUTTON: enables focus point movement in X-Y-Plane." << std::endl;
  std::cout << "ALT + CTRL + MIDDLE_BUTTON: enables focus point movement in Z-direction." << std::endl;
  std::cout << "ALT + CTRL + MOUSE_WHEEL: Move Camera closer of further away from focus point." << std::endl;
  std::cout << "MOUSE_WHEEL: increase/ decrease super voxel size." << std::endl;
  std::cout << "" << std::endl;
}

void Visualizer::log()
{
  if (m_log_file.empty())
  {
    logCreate();
  }

  uint32_t num_voxels = 0;
  for (uint32_t i = 0; i < m_cur_context->m_voxel_maps.size(); i++)
  {
    num_voxels += m_cur_context->m_voxel_maps[i]->m_total_num_voxels;
  }
  std::ofstream log_file(m_log_file.c_str(), std::ios_base::out | std::ios_base::app);
  std::string mode;
  if (m_cur_context->m_draw_whole_map)
    mode = "whole";
  else
    mode = "view";

  log_file << num_voxels << "\t" << 36 * num_voxels << "\t" << m_cur_fps << "\t" << mode << "\t"
      << m_cur_context->m_dim_svoxel.x << std::endl;

  log_file.close();
}

void Visualizer::logCreate()
{
//create a log file with the current time in the name

  time_t rawtime;
  struct tm * timeinfo;
  char buffer[80];

  time(&rawtime);
  timeinfo = localtime(&rawtime);

  strftime(buffer, 80, "%F_%H.%M.%S", timeinfo);
  std::string t = buffer;
  m_log_file = "benchmarks/benchmark_" + t + ".log";

  std::cout << "Creating a new log file:   " << m_log_file << std::endl;

  std::ofstream log_file(m_log_file.c_str(), std::ios_base::out | std::ios_base::app);

  log_file << "Benchmark: " << std::endl << std::endl;
  log_file << "Dimension of voxel map x: y: z: side length: " << std::endl;
  log_file << "Number of inserted Voxels:" << std::endl;
  log_file << std::endl;
  log_file << "#Voxels  #Triangles  fps  drawMode   svoxelSize " << std::endl;
  log_file << std::endl;

  log_file.close();
}






} // end of namespace visualization
} // end of namespace gpu_voxels
