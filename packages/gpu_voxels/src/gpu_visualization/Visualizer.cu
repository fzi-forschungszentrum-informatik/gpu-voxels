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

#include <gpu_visualization/shaders/ColormapFragmentShader.h>
#include <gpu_visualization/shaders/ColormapVertexShader.h>
#include <gpu_visualization/shaders/LightingFragmentShader.h>
#include <gpu_visualization/shaders/LightingVertexShader.h>
#include <gpu_visualization/shaders/SimpleFragmentShader.h>
#include <gpu_visualization/shaders/SimpleVertexShader.h>

namespace gpu_voxels {
namespace visualization {

namespace bfs = boost::filesystem;
using namespace glm;

Visualizer::Visualizer()
{
  setMaxMem(0);
  m_cur_mem = 0;
  m_d_positions = NULL;
  m_number_of_primitives = 0;
  m_type_primitive = primitive_INITIAL_VALUE;
  m_vbo_primitives_pos = 0;
  m_default_prim = NULL;
  m_shm_manager_octrees = NULL;
  m_shm_manager_voxelmaps = NULL;
  m_shm_manager_visualizer = NULL;
  m_window_title = "visualizer";
}

Visualizer::~Visualizer()
{
  delete m_cur_context;
  delete m_interpreter;
  delete m_default_prim;
  delete m_shm_manager_octrees;
  delete m_shm_manager_visualizer;
  delete m_shm_manager_voxelmaps;
  for (std::vector<Primitive*>::iterator it = m_primitives.begin(); it != m_primitives.end(); ++it)
  {
    delete *it;
  }
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
  glutInitContextVersion(4, 3);
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

  // Create and compile the GLSL program from the shaders bfs = boost::filesystem
  char* path_home;
  path_home = getenv("MCAHOME");
  bfs::path full_path;
  if (path_home != NULL)
  {
    full_path = bfs::path(path_home) / "build";
  }
  else
  {
    LOGGING_INFO_C(Visualization, Visualizer,
                   "Environment variables not set (MCAHOME)! The default path is used instead." << endl);
    full_path = bfs::path(bfs::initial_path<bfs::path>());
    full_path = bfs::system_complete(bfs::path(argv[0]));
    full_path.remove_filename();
    if (full_path.leaf() == ".")
    {
      full_path.remove_leaf();
    }
    full_path.remove_leaf();
  }
  full_path /= "shader";
  LOGGING_INFO_C(Visualization, Visualizer, "Search path of the shaders is: " << full_path.c_str() << endl);

  //bfs::path vertex_shader_path = full_path / "SimpleVertexShader.vertexshader";
  //bfs::path fragment_shader_path = full_path / "SimpleFragmentShader.fragmentshader";
  m_programID = loadShaders(SimpleVertexShader::get(), SimpleFragmentShader::get());

  //vertex_shader_path = full_path / "colormap.vertexshader";
  //fragment_shader_path = full_path / "colormap.fragmentshader";
  m_colormap_programID = loadShaders(ColormapVertexShader::get(), ColormapFragmentShader::get());

  //vertex_shader_path = full_path / "lighting.vertexshader";
  //fragment_shader_path = full_path / "lighting.fragmentshader";
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

bool Visualizer::initializeContextFromXML(int& argc, char *argv[])
{
  m_interpreter = new XMLInterpreter();
  m_interpreter->initialize(argc, argv);

  m_max_mem = m_interpreter->getMaxMem();
  m_max_fps = m_interpreter->getMaxFps();
  m_interpreter->getPrimtives(m_primitives);

  m_cur_context = new VisualizerContext();
  bool suc = m_interpreter->getVisualizerContext(m_cur_context);
  copyDrawTypeToDevice();
  return suc;
}

bool Visualizer::initalizeVisualizer(int& argc, char *argv[])
{
  m_shm_manager_visualizer = new SharedMemoryManagerVisualizer();
  return initializeContextFromXML(argc, argv) & initGL(&argc, argv);
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
bool Visualizer::resizeGLBufferForOctree(OctreeContext* con)
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
    //  cuPrintDeviceMemoryInfo();
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
  //  cuPrintDeviceMemoryInfo();
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
    glm::vec3 p2 = glm::vec3(i, 0, y_max);
    data.push_back(p1);
    data.push_back(p2);
  }
  for (float i = 0; i <= y_max; i += distance)
  {/*Insert the lines along the z axis*/
    glm::vec3 p1 = glm::vec3(0, 0, i);
    glm::vec3 p2 = glm::vec3(x_max, 0, i);
    data.push_back(p1);
    data.push_back(p2);
  }
  size_t size = 2.f * ((y_max + x_max) / distance + 2); // two points per line and one line for each x and z value
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

void Visualizer::createSphere(glm::vec3 pos, float radius, glm::vec4 color)
{
  Sphere* s = new Sphere(color, pos, radius, 16);
  s->create(m_cur_context->m_lighting);
  m_primitives.push_back(s);
}

void Visualizer::createCuboid(glm::vec3 pos, glm::vec3 side_length, glm::vec4 color)
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

  OctreeContext* con = new OctreeContext(map_name);
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
    m_cur_context->m_camera->setCameraTarget(vec3(d.x / 2.f, 0.f, d.y / 2.f));
    m_cur_context->m_camera->setCameraTargetOfInitContext(vec3(d.x / 2.f, 0.f, d.y / 2.f));
    createFocusPointVBO();
  }
  m_cur_context->m_voxel_maps.push_back(con);
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
  if (context->m_voxelMap->getMapType() == MT_BIT_VOXELMAP)
  {
    if(voxelmap::BIT_VECTOR_LENGTH > MAX_DRAW_TYPES)
      LOGGING_ERROR_C(Visualization, Visualizer,
          "Only " << MAX_DRAW_TYPES << " different draw types supported. But bit vector has " << voxelmap::BIT_VECTOR_LENGTH << " different types." << endl);

    fill_vbo_without_precounting<<<context->m_num_blocks, context->m_threads_per_block>>>(
        /**/
        (voxelmap::BitVectorVoxel*) context->m_voxelMap->getVoidDeviceDataPtr(),/**/
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

  }
  else if(context->m_voxelMap->getMapType() == MT_PROBAB_VOXELMAP)
  {
    fill_vbo_without_precounting<<<context->m_num_blocks, context->m_threads_per_block>>>(
        /**/
        (voxelmap::ProbabilisticVoxel*) context->m_voxelMap->getVoidDeviceDataPtr(),/**/
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
  thrust::host_vector<uint32_t> temp = context->m_num_voxels_per_type;
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

void Visualizer::fillGLBufferWithOctree(OctreeContext* context, uint32_t index)
{
  if (!updateOctreeContext(context, index))
  {/*Don't fill the vbo if it couldn't be updated.*/
    return;
  }

  thrust::device_vector<uint32_t> indices(context->m_num_voxels_per_type.size(), 0);
  calculateNumberOfCubeTypes(context);

  context->m_vbo_draw_able = resizeGLBufferForOctree(context);
  if (context->m_vbo_draw_able)
  {
    float4 *vbo_ptr;
    size_t num_bytes;
    HANDLE_CUDA_ERROR(cudaGraphicsMapResources(1, &(context->m_cuda_ressources), 0));
    HANDLE_CUDA_ERROR(
        cudaGraphicsResourceGetMappedPointer((void ** )&vbo_ptr, &num_bytes, context->m_cuda_ressources));

    // Launch kernel to copy data into the OpenGL buffer.
    fill_vbo_with_octree<<<context->m_num_blocks, context->m_threads_per_block>>>(
        /**/
        context->getCubesDevicePointer(),/**/
        context->getNumberOfCubes(),/**/
        vbo_ptr,/**/
        thrust::raw_pointer_cast(context->m_d_vbo_offsets.data()),/**/
        thrust::raw_pointer_cast(indices.data()),/**/
        thrust::raw_pointer_cast(m_cur_context->m_d_draw_types.data()),/**/
        thrust::raw_pointer_cast(m_cur_context->m_d_prefixes.data()));/**/

    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    HANDLE_CUDA_ERROR(cudaGraphicsUnmapResources(1, &context->m_cuda_ressources, 0));
    cudaIpcCloseMemHandle(context->getCubesDevicePointer());
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

void Visualizer::calculateNumberOfCubeTypes(OctreeContext* context)
{
  thrust::fill(context->m_d_num_voxels_per_type.begin(), context->m_d_num_voxels_per_type.end(), 0);
// Launch kernel to copy data into the OpenGL buffer. <<<context->getNumberOfCubes(),1>>><<<num_threads_per_block,num_blocks>>>
  calculate_cubes_per_type<<<context->m_num_blocks, context->m_threads_per_block>>>(
      context->getCubesDevicePointer(),/**/
      context->getNumberOfCubes(),/**/
      thrust::raw_pointer_cast(context->m_d_num_voxels_per_type.data()),
      thrust::raw_pointer_cast(m_cur_context->m_d_draw_types.data()),/**/
      thrust::raw_pointer_cast(m_cur_context->m_d_prefixes.data()));/**/

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  context->m_num_voxels_per_type = context->m_d_num_voxels_per_type;
  context->updateVBOOffsets();
  context->updateTotalNumVoxels();
}

void Visualizer::updateStartEndViewVoxelIndices()
{
  if (m_cur_context->m_draw_whole_map)
  {
    m_cur_context->m_view_start_voxel_pos = m_cur_context->m_min_xyz_to_draw;
    m_cur_context->m_view_end_voxel_pos = m_cur_context->m_max_xyz_to_draw;
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

bool Visualizer::updateOctreeContext(OctreeContext* context, uint32_t index)
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
  float v = 256.f * 256.f * pixel[0]/*r*/+ 256.f * pixel[1]/*g*/+ pixel[2]/*b*/;
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
      if (context->m_types_segment_mapping[i] == eVT_COLLISION)
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
        * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, m_cur_context->m_grid_height, 0.f));

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
    glUniform4f(m_startColorID, 1, 0, 0, 1); // draw the x-axis with red
    glDrawArrays(GL_LINES, 0, 2);
    glUniform4f(m_startColorID, 0, 1, 0, 1); // draw the y-axis with green
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
    glDrawArrays(GL_POINTS, 0, 1);
  }
}

void Visualizer::drawPrimitives()
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
// only read the buffer again if it has changed
  if (m_shm_manager_visualizer != NULL && m_shm_manager_visualizer->hasPrimitiveBufferChanged())
  {
    PrimitiveTypes old_type = m_type_primitive;
    if (m_shm_manager_visualizer->getPrimitivePositions(m_d_positions, m_number_of_primitives,
                                                        m_type_primitive))
    {
// generate new buffer after deleting the old one.
      glDeleteBuffers(1, &m_vbo_primitives_pos);
      glGenBuffers(1, &m_vbo_primitives_pos);
      glBindBuffer(GL_ARRAY_BUFFER, m_vbo_primitives_pos);
      glBufferData(GL_ARRAY_BUFFER, m_number_of_primitives * SIZE_OF_TRANSLATION_VECTOR, 0, GL_STATIC_DRAW);
      glBindBuffer(GL_ARRAY_BUFFER, 0);

// copy the data from the list into the OpenGL buffer
      struct cudaGraphicsResource* cuda_res;
      HANDLE_CUDA_ERROR(
          cudaGraphicsGLRegisterBuffer(&cuda_res, m_vbo_primitives_pos,
                                       cudaGraphicsRegisterFlagsWriteDiscard));
      glm::vec4 *vbo_ptr;
      size_t num_bytes;
      HANDLE_CUDA_ERROR(cudaGraphicsMapResources(1, &(cuda_res), 0));
      HANDLE_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer((void ** )&vbo_ptr, &num_bytes, cuda_res));

      cudaMemcpy(vbo_ptr, m_d_positions, m_number_of_primitives * SIZE_OF_TRANSLATION_VECTOR,
                 cudaMemcpyDeviceToDevice);

      cudaIpcCloseMemHandle(m_d_positions);
      HANDLE_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cuda_res, 0));
      HANDLE_CUDA_ERROR(cudaGraphicsUnregisterResource(cuda_res));
// update the changed variable in the shared memory
      m_shm_manager_visualizer->setPrimitiveBufferChangedToFalse();
    }
    else
      // if it was not possible to load data from shared memory
      return;

// if the used type has changed generate the corresponding default primitive
    if (old_type != m_type_primitive)
    {
      if (m_type_primitive == primitive_Sphere)
      {
        Sphere* sphere;
        m_interpreter->getDefaultSphere(sphere);
        m_default_prim = sphere;
        // m_default_prim = new Sphere(glm::vec4(1, 0, 0, 1), glm::vec3(0, 0, 0), 1.f, 16);
        m_default_prim->create(m_cur_context->m_lighting);
      }
      else if (m_type_primitive == primitive_Cuboid)
      {
        Cuboid* cuboid;
        m_interpreter->getDefaultCuboid(cuboid);
        m_default_prim = cuboid;
        //m_default_prim = new Cuboid(glm::vec4(1, 0, 0, 1), glm::vec3(0, 0, 0), glm::vec3(1.f));
        m_default_prim->create(m_cur_context->m_lighting);
      }
      else
      {
        LOGGING_WARNING_C(Visualization, Visualizer,
                          "Primitive type not supported yet ... add it here." << endl);

      }
    }
  }
  if (m_default_prim != NULL)
  {
    GLuint color_id;
    mat4 V = m_cur_context->m_camera->getViewMatrix();
    mat4 VP = m_cur_context->m_camera->getProjectionMatrix() * V;
    if (m_cur_context->m_lighting)
    { // set up the correct variables for the shader with lighting
      glUseProgram(m_lighting_programID);
      mat4 V_inv_trans = transpose(inverse(V));
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
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo_primitives_pos);
    glEnableVertexAttribArray(2);
    glVertexAttribDivisor(2, 1);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*) 0);

    if (m_cur_context->m_draw_filled_triangles)
    {
      glUniform4fv(color_id, 1, glm::value_ptr(m_default_prim->getColor()));
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
      m_default_prim->draw(m_number_of_primitives, m_cur_context->m_lighting);
    }
    ExitOnGLError("ERROR! Couldn't draw the filled triangles of the primitives.");
    if (m_cur_context->m_draw_edges_of_triangels)
    {
      glPolygonOffset(-1.f, -1.f);
      glEnable(GL_POLYGON_OFFSET_LINE);
      glm::vec4 c = m_default_prim->getColor() * 0.5f;
      glUniform4fv(color_id, 1, glm::value_ptr(c));
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
      m_default_prim->draw(m_number_of_primitives, m_cur_context->m_lighting);
      glDisable(GL_POLYGON_OFFSET_LINE);
    }
    ExitOnGLError("ERROR! Couldn't draw the edges of the primitives.");

  }
// reset used stuff...
  glVertexAttribDivisor(2, 0);
  glDisableVertexAttribArray(2);
  glUseProgram(0);

}

///////////////////////////////Begin: Callback functions for freeglut///////////////////////////////
void Visualizer::renderFunction(void)
{
  m_frameCount++;
  int32_t lastTime = glutGet(GLUT_ELAPSED_TIME);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

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
  drawPrimitives();

// set some of the unfirom shader variables for the data contexts
  mat4 VP = m_cur_context->m_camera->getProjectionMatrix() * m_cur_context->m_camera->getViewMatrix();
  if (m_cur_context->m_lighting)
  {
    glUseProgram(m_lighting_programID);
    mat4 V = m_cur_context->m_camera->getViewMatrix();
    mat4 V_inv_trans = transpose(inverse(V));
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

  if (m_shm_manager_visualizer != NULL)
  {
    DrawTypes d = m_shm_manager_visualizer->getDrawTypes();
    bool changed = false;
    for(uint32_t i = eVT_SWEPT_VOLUME_START + 1; i < MAX_DRAW_TYPES; ++i)
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

////////////////////////////////draw all octrees ///////////////////////////////////////

  for (uint32_t i = 0; i < m_cur_context->m_octrees.size(); i++)
  {
    if (m_cur_context->m_octrees[i]->m_draw_context)
    {
      if (m_shm_manager_octrees->hasOctreeBufferSwapped(i))
      {
        fillGLBufferWithOctree(m_cur_context->m_octrees[i], i);
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
    float wait = 1000.f / m_max_fps - m_delta_time;
    m_delta_time += wait;
    boost::this_thread::sleep(boost::posix_time::milliseconds(wait));
  }
}

void Visualizer::mouseClickFunction(int32_t button, int32_t state, int32_t xpos, int32_t ypos)
{
  if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN)
  {
    printPositionOfVoxelUnderMouseCurser(xpos, ypos);
  }
  else if (button == 3 && state == GLUT_DOWN) /*mouse wheel up*/
  {
    increaseSuperVoxelSize();
  }
  else if (button == 4 && state == GLUT_DOWN)/*mouse wheel down*/
  {
    decreaseSuperVoxelSize();
  }

}

/*
 * The xpos and ypos parameters indicate the mouse location in window relative coordinates.
 */
void Visualizer::mouseMotionFunction(int32_t xpos, int32_t ypos)
{

  int32_t modi = glutGetModifiers();
  if (modi == (GLUT_ACTIVE_CTRL | GLUT_ACTIVE_ALT))
  {
    m_cur_context->m_camera->moveFocusPointFromMouseInput(xpos, ypos);
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
    s << m_window_title << ": " << m_frameCount * 4 << " Frames Per Second @ " << getWidth() << " x "
        << getHeight() << " Super Voxel Size: " << m_cur_context->m_dim_svoxel.x;
    glutSetWindowTitle(s.str().c_str());

    m_cur_fps = m_frameCount;
  }
  m_frameCount = 0;
  glutTimerFunc(250, callback, 1);
}

void Visualizer::keyboardFunction(unsigned char key, int32_t x, int32_t y)
{
  float multiplier = 1.f;
  if (glutGetModifiers() == GLUT_ACTIVE_SHIFT)
  {
    multiplier *= 5.f;
  }
  else if (glutGetModifiers() == GLUT_ACTIVE_CTRL)
  {
    multiplier *= 10.f;
  }

  // size_t temp = 0;
  switch (key)
  {
    case 'b':
      printNumberOfVoxelsDrawn();
      break;
    case 'c': // toggle camera mode
      m_cur_context->m_camera->toggleCameraMode();
      break;
    case 'd': //Toggle draw grid
      m_cur_context->m_draw_grid = !m_cur_context->m_draw_grid;
      break;
    case 'e': //Toggle draw edges
      m_cur_context->m_draw_edges_of_triangels = !m_cur_context->m_draw_edges_of_triangels;
      break;
    case 'f': //Toggle fill triangles
      m_cur_context->m_draw_filled_triangles = !m_cur_context->m_draw_filled_triangles;
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
      cuPrintDeviceMemoryInfo();
      break;
    case 'p':
      m_cur_context->m_camera->printCameraPosDirR();
      break;
    case 'r': // reset camera
      m_cur_context->m_camera->resetToInitialValues();
      createFocusPointVBO();
      break;
    case 't':
      m_cur_context->m_camera->printCameraTargetPointPos();
      break;
    case 'v':
      printViewInfo();
      break;
    case 'x':
      toggleLighting();
      break;

    case '+': // increase light intensity
      m_cur_context->m_light_intensity += 20.f * multiplier;
      break;
    case '-': // decrease light intensity
      m_cur_context->m_light_intensity -= 20.f * multiplier;
      m_cur_context->m_light_intensity = max(m_cur_context->m_light_intensity, 0.f);
      break;
    case '1':
      flipDrawType(eVT_OCCUPIED);
      break;
    case '2':
      flipDrawType(eVT_COLLISION);
      break;
    case '3':
      flipDrawType(eVT_SWEPT_VOLUME_START);
      break;
    case '4':
      flipDrawType(VoxelType(eVT_SWEPT_VOLUME_START + 1));
      break;
    case '5':
      flipDrawType(VoxelType(eVT_SWEPT_VOLUME_START + 2));
      break;
    case '6':
      flipDrawType(VoxelType(eVT_SWEPT_VOLUME_START + 3));
      break;
    case '7':
      flipDrawType(VoxelType(eVT_SWEPT_VOLUME_START + 4));
      break;
    case '8':
      flipDrawType(VoxelType(eVT_SWEPT_VOLUME_START + 5));
      break;
    case '9':
      flipDrawType(eVT_FREE);
      break;
    case '0':
      flipDrawType(eVT_UNKNOWN);
      break;

  }
}

void Visualizer::keyboardSpecialFunction(int32_t key, int32_t x, int32_t y)
{
  float speed_multiplier = m_cur_context->m_dim_svoxel.x * 0.3f;

  if (glutGetModifiers() == GLUT_ACTIVE_SHIFT)
  {
    speed_multiplier *= 10.f;
  }
  else if (glutGetModifiers() == GLUT_ACTIVE_CTRL)
  {
    speed_multiplier *= 30.f;
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
      flipDrawVoxelmap(0);
      break;
    case GLUT_KEY_F2:
      flipDrawVoxelmap(1);
      break;
    case GLUT_KEY_F3:
      flipDrawVoxelmap(2);
      break;
    case GLUT_KEY_F4:
      flipDrawVoxelmap(3);
      break;
    case GLUT_KEY_F5:
      flipDrawOctree(0);
      break;
    case GLUT_KEY_F6:
      flipDrawOctree(1);
      break;
    case GLUT_KEY_F7:
      flipDrawOctree(2);
      break;
    case GLUT_KEY_F8:
      flipDrawOctree(3);
      break;
    default:
      break;
  }
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
    d = d / 2;
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
    LOGGING_INFO_C(Visualization, Visualizer, "No voxel map registered at index " << index << endl);
    return;
  }
  VoxelmapContext* con = m_cur_context->m_voxel_maps[index];
  con->m_draw_context = !con->m_draw_context;

  if (con->m_draw_context)
  {
    m_cur_context->m_camera->setViewChanged(true); // force an update of vbos
    LOGGING_INFO_C(Visualization, Visualizer,
                   "Activated the drawing of the voxel map: " << con->m_map_name << endl);
  }
  else
  {
    LOGGING_INFO_C(Visualization, Visualizer,
                   "Deactivated the drawing of the voxel map: " << con->m_map_name << endl);
  }
}
void Visualizer::flipDrawOctree(uint32_t index)
{
  if (index >= m_cur_context->m_octrees.size())
  {
    LOGGING_INFO_C(Visualization, Visualizer, "No octree registered at index " << index << endl);
    return;
  }
  OctreeContext* con = m_cur_context->m_octrees[index];
  con->m_draw_context = !con->m_draw_context;
  if (con->m_draw_context)
  {
    LOGGING_INFO_C(Visualization, Visualizer,
                   "Activated the drawing of the octree: " << con->m_map_name << endl);
  }
  else
  {
    LOGGING_INFO_C(Visualization, Visualizer,
                   "Deactivated the drawing of the octree: " << con->m_map_name << endl);
  }
}

void Visualizer::flipDrawType(VoxelType type)
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
 * Prints the position of the voxel under the mouse curser.
 */
void Visualizer::printPositionOfVoxelUnderMouseCurser(int32_t xpos, int32_t ypos)
{
//clear background color
  glClearColor(1.f, 1.f, 1.f, 1.f);
// if x or y or z is equal to 2^24 - 1 the clicked point should be the background
  uint32_t background_pos = pow(2.f, 24.f) - 1.f;
  Vector3ui d = m_cur_context->m_dim_svoxel;
  glm::vec3 c_pos = m_cur_context->m_camera->getCameraPosition();
  bool found_in_voxelmap = false;
  bool found_in_octree = false;
  Vector3ui n_pos;
  uint32_t data_index;
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
    OctreeContext * con = m_cur_context->m_octrees[i];
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

//reset background color
  glm::vec4 bc = m_cur_context->m_background_color;
  glClearColor(bc.x, bc.y, bc.z, 1.f);
  std::cout << std::endl;
  if (!found_in_voxelmap && !found_in_octree)
  {
    std::cout << "There is no voxel!" << std::endl;
    return;
  }
  float scale = m_cur_context->m_scale_unit.first;
  std::string unit = m_cur_context->m_scale_unit.second;
  if (found_in_voxelmap)
  {
    std::cout << "Found voxel in voxel map: " << m_cur_context->m_voxel_maps[data_index]->m_map_name
        << std::endl;
    n_pos = n_pos / d;

    if (d.x > 1 || d.y > 1 || d.z > 1)
    {
// if super voxel dim > 0, print the start end end voxel
      n_pos = n_pos * d;
      Vector3ui end_pos = n_pos + d - Vector3ui(1);

      std::cout << "Start voxel position x: " << n_pos.x << " y: " << n_pos.z << " z: " << n_pos.y
          << std::endl;
      std::cout << "Start voxel distance x: " << n_pos.x * scale << unit << " y: " << n_pos.z * scale << unit
          << " z: " << n_pos.y * scale << unit << std::endl;
      std::cout << "End voxel position x: " << end_pos.x << " y: " << end_pos.z << " z: " << end_pos.y
          << std::endl;
      std::cout << "End voxel distance x: " << end_pos.x * scale << unit << " y: " << end_pos.z * scale
          << unit << " z: " << end_pos.y * scale << unit << std::endl;
    }
    else
    {
      std::cout << "Voxel position x: " << n_pos.x << " y: " << n_pos.z << " z: " << n_pos.y << std::endl;
      std::cout << "Voxel distance x: " << n_pos.x * scale << unit << " y: " << n_pos.z * scale << unit
          << " z: " << n_pos.y * scale << unit << std::endl;
    }

  }
  if (found_in_octree)
  {
    std::cout << "Found voxel in octree: " << m_cur_context->m_octrees[data_index]->m_map_name << std::endl;
    std::cout << "Voxel position x: " << n_pos.x << " y: " << n_pos.z << " z: " << n_pos.y << std::endl;
    std::cout << "Voxel distance x: " << n_pos.x * scale << unit << " y: " << n_pos.z * scale << unit
        << " z: " << n_pos.y * scale << unit << std::endl;
  }
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
  /*eVT_SWEPT_VOLUME_START(0) -> 254 */
  //return type == (uint8_t) 255 ? (uint8_t) 255 : (uint8_t) 254 - type;
  return type;
}
/**
 * Distributes the maximum usable memory on all voxel maps
 */
void Visualizer::distributeMaxMemory()
{
  uint32_t num_data_structures = m_cur_context->m_voxel_maps.size() + m_cur_context->m_octrees.size();

  if (num_data_structures == 0)
  {
    return;
  }
  size_t mem_per_data_structure = m_max_mem / num_data_structures;

  for (uint32_t i = 0; i < m_cur_context->m_voxel_maps.size(); i++)
  {
    m_cur_context->m_voxel_maps[i]->m_max_vbo_size = mem_per_data_structure;
  }

  for (uint32_t i = 0; i < m_cur_context->m_octrees.size(); i++)
  {
    m_cur_context->m_octrees[i]->m_max_vbo_size = mem_per_data_structure;
  }

}

void Visualizer::printNumberOfVoxelsDrawn()
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
  std::cout << "#Voxels " << num_voxels << " #Triangles " << num_voxels * 36 << std::endl;
}

void Visualizer::printTotalVBOsizes()
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
  std::cout << "Current Size of the VBOs: " << total_size / 1e+006 << " MByte.\n";
}

void Visualizer::printViewInfo()
{
  std::cout << "--------printing view info--------" << std::endl;
  glm::vec3 vd = m_cur_context->m_dim_view;
  std::cout << "dimension of view x: " << vd.x << " y: " << vd.y << " z: " << vd.z << std::endl;

  Vector3ui vs = m_cur_context->m_view_start_voxel_pos;
  std::cout << "m_view_start_voxel_index x: " << vs.x << " y: " << vs.y << " z: " << vs.z << std::endl;
  Vector3ui ve = m_cur_context->m_view_end_voxel_pos;
  std::cout << "m_view_end_voxel_index x: " << ve.x << " y: " << ve.y << " z: " << ve.z << std::endl;

  glm::vec3 d = m_cur_context->m_camera->getCameraDirection();
  std::cout << "Camera direction  x: " << d.x << " y: " << d.y << " z: " << d.z << std::endl;

  glm::vec3 t = m_cur_context->m_camera->getCameraTarget();
  std::cout << "Camera target: x: " << t.x << " y: " << t.y << " z: " << t.z << std::endl;
}

/**
 * Prints the key bindings.
 */
void Visualizer::printHelp()
{
  std::cout << "" << std::endl;
  std::cout << "" << std::endl;
  std::cout << "Visualizer Controls" << std::endl;
  std::cout << "" << std::endl;
  std::cout << "---->Keyboard" << std::endl;
  std::cout << "h: print this help." << std::endl;
  std::cout << "v: print view info." << std::endl;
  std::cout << "p: print camera position." << std::endl;
  std::cout << "m: print total VBO size." << std::endl;
  std::cout << "n: print device memory info." << std::endl;
  std::cout << "" << std::endl;
  std::cout << "g: toggle draw whole map." << std::endl;
  std::cout << "d: toggle draw grid." << std::endl;
  std::cout << "e: toggle draw edges of the triangles." << std::endl;
  std::cout << "f: toggle draw filled triangles." << std::endl;
  std::cout << "l|x: toggle lighting on/off." << std::endl;
  std::cout
      << "i: toggle the OpenGL depth function for the collision type. GL_ALWAYS should be used to get an overview (produces artifacts)."
      << std::endl;
  std::cout << "c: toggle between orbit and free-flight camera." << std::endl;
  //std::cout << "k: toggle use of camera target point from shared memory." << std::endl;
  std::cout << "+/-: increase/decrease the light intensity. " << std::endl;
  std::cout << "r: reset camera to default position." << std::endl;
  std::cout << "CRTL: high movement speed." << std::endl;
  std::cout << "SHIFT: medium movement speed." << std::endl;
  std::cout << "0-9: toggle the drawing of the some types." << std::endl;
  std::cout << "F1-F4: toggle the drawing of the first 4 registered voxel maps." << std::endl;
  std::cout << "F5-F8: toggle the drawing of the first 4 registered octrees." << std::endl;
  std::cout << "" << std::endl;
  std::cout << "---->Mouse" << std::endl;
  std::cout << "RIGHT_BUTTON: print x,y,z coordinates of the clicked voxel." << std::endl;
  std::cout << "LEFT_BUTTON: enables mouse movement." << std::endl;
  std::cout << "ALT + CTRL + LEFT_BUTTON: enables focus point movement in X-Y-Plane." << std::endl;
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
