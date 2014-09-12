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
 * \author  Sebastian Klemm
 * \date    2012-12-19
 *
 *  this tool provides tests for the VoxelMap,
 *  RobotMap and EnvironmentMap classes.
 *
 *  Use with map of at least 10x10x10 voxels
 */
//----------------------------------------------------------------------

#include <cstdio>

#include <gpu_voxels/voxelmap/EnvironmentMap.h>
#include <gpu_voxels/voxelmap/RobotMap.h>
#include <gpu_voxels/robot/KinematicChain.h>
#include <gpu_voxels/visualization/VoxelMapVisualizer.h>

using namespace gpu_voxels;
using namespace gpu_voxels::visualization;

VoxelMapVisualizer* vis = new VoxelMapVisualizer();

/* ----- VoxelMap callback wrappers ----- */
void VoxelMapVisualizer_display_wrapper()
{
  vis->display();
}

void VoxelMapVisualizer_reshape_wrapper(int x, int y)
{
  vis->reshape(x, y);
}

void VoxelMapVisualizer_keyboardActions_wrapper(unsigned char key, int x, int y)
{
  vis->keyboardActions(key, x, y);
}

void VoxelMapVisualizer_mouseClicks_wrapper(int button, int state, int x, int y)
{
  vis->mouseClicks(button, state, x, y);
}

void VoxelMapVisualizer_mouseMotion_wrapper(int x, int y)
{
  vis->mouseMotion(x, y);
}

/* ----- VoxelMap run function ----- */
void run_visualization(uint32_t dim_x, uint32_t dim_y, uint32_t dim_z)
{
  /* -- set scene dimensions -- */

  // transformed from World to OpenGL CS:  y, z, x
  vis->getConfig().scene_dim_low  = GLVector3f(0, 0, 0);
  vis->getConfig().scene_dim_high = GLVector3f(dim_y, dim_z, dim_x);

  int argc = 0;
  char** argv = NULL;
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);
  glutInitWindowSize(vis->getConfig().window_size.x, vis->getConfig().window_size.y);
  // glutInitWindowSize(800, 600);

  glutCreateWindow("Voxelmap Visualization");

  // vis->initShadingAndLighting();
  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_TRUE);
  glClearColor(0, 0, 0, 0);

  // callbacks
  // todo: implement callbacks via boost shared ptr
  glutDisplayFunc(VoxelMapVisualizer_display_wrapper);
  glutReshapeFunc(VoxelMapVisualizer_reshape_wrapper);
  glutKeyboardFunc(VoxelMapVisualizer_keyboardActions_wrapper);
  glutSetKeyRepeat(GLUT_KEY_REPEAT_ON);
  glutMouseFunc(VoxelMapVisualizer_mouseClicks_wrapper);
  glutMotionFunc(VoxelMapVisualizer_mouseMotion_wrapper);


  vis->display(); // render once to apply lighting etc
  glutMainLoop();
}


int main( int argc, char* argv[] )
{

  printf("Voxelmap test program.\n\n");

  // read in parameters from terminal
    uint32_t dim_x, dim_y, dim_z;
  float voxel_side_length;
  if (argc < 5)
  {
    printf("\nusage: %s dim_x dim_y dim_z voxel_side_length\n", argv[0]);
    exit(-1);
  }
  else
  {
    dim_x = atoi(argv[1]);
    dim_y = atoi(argv[2]);
    dim_z = atoi(argv[3]);
    voxel_side_length = atof(argv[4]);
  }

  printf("\nCreating CUDA VoxelMap(s) of size:\n");
  printf(" - dim x             : %u\n", dim_x);
  printf(" - dim y             : %u\n", dim_y);
  printf(" - dim z             : %u\n", dim_z);
  printf(" - voxel_side_length : %f\n", voxel_side_length);
  printf("--------------------------------\n\n");

  if ((dim_x <10) || (dim_y <10) || (dim_z <10))
  {
    printf("Use at least a map of 10x10x10 voxels for this evaluation!\n");
    exit(-1);
  }
  char key;

  RobotMap* robot_map = new RobotMap(dim_x, dim_y, dim_z, voxel_side_length);
  EnvironmentMap* env_map = new EnvironmentMap(dim_x, dim_y, dim_z, voxel_side_length);

  // register maps for visualization
  vis->registerRobotMap(robot_map);
  vis->registerEnvironmentMap(env_map);

  // split off a thread to run independent from visualization
  boost::thread vis_thread(run_visualization, dim_x, dim_y, dim_z);


  //  ================== TESTS ==================
  const uint8_t col_check_threshold = 100;
  bool collision  = false;


  // TRUE = test passed, FALSE = test failed
  std::vector<bool> test_results;


  /* -------------------------------------------------------------------------------------------------------------
   *  Test: Avoiding collisions
   * -------------------------------------------------------------------------------------------------------------
   */
  printf("\n\n Test %u:\n", test_results.size()+1);
  printf("\n--- Collision check that should NOT cause a collision ---\n");
  env_map->insertBoxByIndices(Vector3ui(1, 1, 1), Vector3ui(3, 3, 3),
                              Voxel::eC_EXECUTION, Voxel::eVT_OCCUPIED);

  robot_map->insertBoxByIndices(Vector3ui(4, 4, 4), Vector3ui(8, 8, 8),
                                Voxel::eC_EXECUTION, Voxel::eVT_OCCUPIED);

  printf("Collision check...\n");
  collision = env_map->collisionCheck(col_check_threshold, robot_map, col_check_threshold, Voxel::eC_EXECUTION);

  printf("There was %s collision.\n", (collision? "a" : "no"));

  test_results.push_back(!collision);

  printf("Press key to proceed to next test.\n");
  std::cin >> key;

  printf("Clearing voxelmaps.. ");
  env_map->clearVoxelMap(Voxel::eC_EXECUTION);
  robot_map->clearVoxelMap(Voxel::eC_EXECUTION);
  printf(" .. done\n");

  //  ---

  printf("\n\n Test %u:\n", test_results.size()+1);
  printf("\n--- Collision check that should NOT cause a collision ---\n");
  env_map->insertBoxByIndices(Vector3ui(0, 0, 0), Vector3ui(5, 5, 5),
                              Voxel::eC_EXECUTION, Voxel::eVT_OCCUPIED);

  robot_map->insertBoxByIndices(Vector3ui(8, 8, 8), Vector3ui(9, 9, 9),
                                Voxel::eC_EXECUTION, Voxel::eVT_OCCUPIED);

  printf("Collision check...\n");
  collision = env_map->collisionCheck(col_check_threshold, robot_map, col_check_threshold, Voxel::eC_EXECUTION);

  printf("There was %s collision.\n", (collision? "a" : "no"));

  test_results.push_back(!collision);

  printf("Press key to proceed to next test.\n");
  std::cin >> key;

  printf("Clearing voxelmaps.. ");
  env_map->clearVoxelMap(Voxel::eC_EXECUTION);
  robot_map->clearVoxelMap(Voxel::eC_EXECUTION);
  printf(" .. done\n");

  //  ---

  printf("\n\n Test %u:\n", test_results.size()+1);
  printf("\n--- Collision check that should NOT cause a collision ---\n");
  env_map->insertBoxByIndices(Vector3ui(1, 3, 3), Vector3ui(2, 3, 3),
                              Voxel::eC_EXECUTION, Voxel::eVT_OCCUPIED);

  robot_map->insertBoxByIndices(Vector3ui(5, 4, 4), Vector3ui(8, 5, 5),
                                Voxel::eC_EXECUTION, Voxel::eVT_OCCUPIED);

  printf("Collision check...\n");
  collision = env_map->collisionCheck(col_check_threshold, robot_map, col_check_threshold, Voxel::eC_EXECUTION);

  printf("There was %s collision.\n", (collision? "a" : "no"));

  test_results.push_back(!collision);

  printf("Press key to proceed to next test.\n");
  std::cin >> key;

  printf("Clearing voxelmaps.. ");
  env_map->clearVoxelMap(Voxel::eC_EXECUTION);
  robot_map->clearVoxelMap(Voxel::eC_EXECUTION);
  printf(" .. done\n");


  /* -------------------------------------------------------------------------------------------------------------
   *  Test: Provoking collisions
   * -------------------------------------------------------------------------------------------------------------
   */
  printf("\n\n Test %u:\n", test_results.size()+1);
  printf("\n--- Collision check that SHOULD cause a collision ---\n");
  env_map->insertBoxByIndices(Vector3ui(1, 1, 1), Vector3ui(4, 4, 4),
                              Voxel::eC_EXECUTION, Voxel::eVT_OCCUPIED);

  robot_map->insertBoxByIndices(Vector3ui(4, 4, 4), Vector3ui(8, 8, 8),
                                Voxel::eC_EXECUTION, Voxel::eVT_OCCUPIED);


  printf("Collision check...\n");
  uint32_t col_count = env_map->collisionCheckWithCounter(col_check_threshold, robot_map, col_check_threshold, Voxel::eC_EXECUTION);
  printf(" ------------ There were %u collisions!\n", col_count);
  if (col_count >0)
  {
    collision = true;
  }

  printf("There was %s collision.\n", (collision? "a" : "no"));

  test_results.push_back(collision);

  printf("Press key to proceed to next test.\n");
  std::cin >> key;


  printf("Clearing voxelmaps.. ");
  env_map->clearVoxelMap(Voxel::eC_EXECUTION);
  robot_map->clearVoxelMap(Voxel::eC_EXECUTION);
  printf(" .. done\n");


  // -------------------------------------------------------------------------------------------------------------
  printf("\n\n Test %u:\n", test_results.size()+1);
  printf("\n--- Collision check that SHOULD cause a collision ---\n");
  env_map->insertBoxByIndices(Vector3ui(4, 5, 6), Vector3ui(7, 8, 9),
                              Voxel::eC_EXECUTION, Voxel::eVT_OCCUPIED);

  robot_map->insertBoxByIndices(Vector3ui(3, 3, 3), Vector3ui(7, 7, 7),
                                Voxel::eC_EXECUTION, Voxel::eVT_OCCUPIED);


  printf("Collision check...\n");
  col_count = env_map->collisionCheckWithCounter(col_check_threshold, robot_map, col_check_threshold, Voxel::eC_EXECUTION);
  printf(" ------------ There were %u collisions!\n", col_count);
  if (col_count >0)
  {
    collision = true;
  }
  printf("There was %s collision.\n", (collision? "a" : "no"));
  test_results.push_back(collision);

  printf("Press key to proceed to next test.\n");
  std::cin >> key;


  printf("Clearing voxelmaps.. ");
  env_map->clearVoxelMap(Voxel::eC_EXECUTION);
  robot_map->clearVoxelMap(Voxel::eC_EXECUTION);
  printf(" .. done\n");


  /* -------------------------------------------------------------------------------------------------------------
   *  Test: inserting artificial sensor data and performing collision checks
   * -------------------------------------------------------------------------------------------------------------
   */

  printf("\n\n Test %u:\n", test_results.size()+1);
  printf("\n--- Inserting artificial sensor data, that should not cause a collision ---\n");
  bool counter_updated = false;

  // sensor pose
  Vector3f sensor_position = Vector3f(10, 50, 50);

  Matrix3f sensor_orientation;
  // (-(pi/2),  0, -(pi/2))
  // [ 0  0  1 ]
  // [-1  0  0 ]
  // [ 0 -1  0 ]

  sensor_orientation.a11 =  0.0;   sensor_orientation.a12 =  0.0;   sensor_orientation.a13 =  1.0;
  sensor_orientation.a21 = -1.0;   sensor_orientation.a22 =  0.0;   sensor_orientation.a23 =  0.0;
  sensor_orientation.a31 =  0.0;   sensor_orientation.a32 = -1.0;   sensor_orientation.a33 =  0.0;

  const uint32_t data_width  = 3;
  const uint32_t data_height = 3;

  Sensor sensor(sensor_position, sensor_orientation, data_width, data_height);
  env_map->initSensorSettings(sensor);

  printf("Sensor pose is (%f, %f, %f, %f, %f, %f)\n",
         sensor_position.x, sensor_position.y, sensor_position.z, -1.5708, 0.0, -1.5708);

  // sensor data
  Vector3f* sensor_data = new Vector3f[sensor.data_size];

  const float sensor_data_distance = 50.0f;

  uint32_t data_index = 0;
  for (uint32_t y=0; y<sensor.data_height; y++)
  {
    for (uint32_t x=0; x<sensor.data_width; x++)
    {
      data_index = y*data_width + x;
      sensor_data[data_index] = Vector3f(x * voxel_side_length - (float)(data_width -1) * 0.5f * voxel_side_length,
                                         y * voxel_side_length - (float)(data_height-1) * 0.5f * voxel_side_length,
                                         sensor_data_distance);

      printf("SensorData(%u) = (%f, %f, %f)\n",
             data_index, sensor_data[data_index].x, sensor_data[data_index].y, sensor_data[data_index].z);
    }
  }
  uint32_t update_counter = env_map->getUpdateCounter();
  env_map->insertSensorData(sensor_data, false, false);
  env_map->increaseUpdateCounter();

  // a) check if update counter did change
  if (update_counter+1 == env_map->getUpdateCounter())
  {
    counter_updated = true;
  }
  else
  {
    counter_updated = false;
  }

  // b) insert data that should not cause collision
  robot_map->insertBoxByIndices(Vector3ui(1, 1, 1), Vector3ui(2, 2, 2),
                                Voxel::eC_EXECUTION, Voxel::eVT_OCCUPIED);

  printf("Collision check...\n");
  collision = env_map->collisionCheck(col_check_threshold, robot_map, col_check_threshold, Voxel::eC_EXECUTION);
  printf("There was %s collision.\n", (collision? "a" : "no"));

  test_results.push_back((!collision && counter_updated));

  printf("Press key to proceed to next test.\n");
  std::cin >> key;


  // ---

  printf("\n\n Test %u:\n", test_results.size()+1);
  printf("\n--- Inserting robot data that SHOULD cause a collision ---\n");

  robot_map->insertBoxByIndices(Vector3ui(3, 3, 3), Vector3ui(8, 8, 8),
                                Voxel::eC_EXECUTION, Voxel::eVT_OCCUPIED);

  printf("Collision check...\n");
  collision = env_map->collisionCheck(col_check_threshold, robot_map, col_check_threshold, Voxel::eC_EXECUTION);
  printf("There was %s collision.\n", (collision? "a" : "no"));

  test_results.push_back(collision);

  printf("Press key to proceed to next test.\n");
  std::cin >> key;

  printf("Clearing voxelmaps.. ");
  env_map->clearVoxelMap(Voxel::eC_EXECUTION);
  robot_map->clearVoxelMap(Voxel::eC_EXECUTION);
  printf(" .. done\n");


  /* -------------------------------------------------------------------------------------------------------------
   *  Test: RayCasting
   * -------------------------------------------------------------------------------------------------------------
   */
  printf("\n\n Test %u:\n", test_results.size()+1);
  printf("\n--- Testing RayCasting ---\n");

  update_counter = env_map->getUpdateCounter();
  env_map->insertSensorData(sensor_data, true, false);
  env_map->increaseUpdateCounter();

  // a) check if update counter did change
  if (update_counter+1 == env_map->getUpdateCounter())
  {
    counter_updated = true;
  }
  else
  {
    counter_updated = false;
  }
  printf("update visualization now.. (\"6\" button in visualization window toggles rays).\n");

  printf("Afterwards press key to proceed .. \n");
  std::cin >> key;
  // b) change sensor-z values (move further away)


  for (uint32_t y=0; y<sensor.data_height; y++)
  {
    for (uint32_t x=0; x<sensor.data_width; x++)
    {
      data_index = y*data_width + x;
      sensor_data[data_index] = Vector3f(x * voxel_side_length - (float)(data_width -1) * 0.5f * voxel_side_length,
                                         y * voxel_side_length - (float)(data_height-1) * 0.5f * voxel_side_length,
                                         sensor_data_distance + 2*voxel_side_length);

      printf("SensorData(%u) = (%f, %f, %f)\n",
             data_index, sensor_data[data_index].x, sensor_data[data_index].y, sensor_data[data_index].z);
    }
  }
  update_counter = env_map->getUpdateCounter();
  env_map->insertSensorData(sensor_data, true, false);
  env_map->increaseUpdateCounter();

  if (update_counter+1 == env_map->getUpdateCounter())
  {
    counter_updated = true;
  }
  else
  {
    counter_updated = false;
  }
  printf("Moving sensor data further in x-direction (red axis) .. This should free previous environment data.\n");
  printf("Update visualization now.. (\"6\" button in visualization window toggles rays)\n");
  printf("Did it work (y/n)? \n");
  std::cin >> key;
  if (key == 'y')
  {
    test_results.push_back(true);
  }

  printf("Clearing voxelmaps.. ");
  env_map->clearVoxelMap(Voxel::eC_EXECUTION);
  robot_map->clearVoxelMap(Voxel::eC_EXECUTION);
  printf(" .. done\n");

  /* -------------------------------------------------------------------------------------------------------------
   *  Test: Cut Robot where whole sensor data is within robot range
   * -------------------------------------------------------------------------------------------------------------
   */
  printf("\n\n Test %u:\n", test_results.size()+1);
  printf("\n--- Testing exclusion of robot from sensor data (w/o RayCasting)---\n");

  printf("Inserting a robot.\n");
    robot_map->insertBoxByIndices(Vector3ui(3, 3, 3), Vector3ui(8, 8, 8),
                                  Voxel::eC_EXECUTION, Voxel::eVT_OCCUPIED);

  for (uint32_t y=0; y<sensor.data_height; y++)
  {
    for (uint32_t x=0; x<sensor.data_width; x++)
    {
      data_index = y*data_width + x;
      sensor_data[data_index] = Vector3f(x * voxel_side_length - (float)(data_width -1) * 0.5f * voxel_side_length,
                                         y * voxel_side_length - (float)(data_height-1) * 0.5f * voxel_side_length,
                                         sensor_data_distance);

      printf("SensorData(%u) = (%f, %f, %f)\n",
             data_index, sensor_data[data_index].x, sensor_data[data_index].y, sensor_data[data_index].z);
    }
  }

  printf("Inserting sensor data, excluding where the robot is.\n");
  update_counter = env_map->getUpdateCounter();
  env_map->insertSensorData(sensor_data, false, true, robot_map->getDeviceDataPtr());
  env_map->increaseUpdateCounter();

  if (update_counter+1 == env_map->getUpdateCounter())
  {
    counter_updated = true;
  }
  else
  {
    counter_updated = false;
  }

  printf("Performing a collision check that should NOT result in a collision.\n");
  printf("Collision check...\n");
  collision = env_map->collisionCheck(col_check_threshold, robot_map, col_check_threshold, Voxel::eC_EXECUTION);
  printf("There was %s collision.\n", (collision? "a" : "no"));

  test_results.push_back((!collision && counter_updated));

  printf("Press key to proceed to next test.\n");
  std::cin >> key;


  printf("Clearing voxelmaps.. ");
  env_map->clearVoxelMap(Voxel::eC_EXECUTION);
  robot_map->clearVoxelMap(Voxel::eC_EXECUTION);
  printf(" .. done\n");

  /* -------------------------------------------------------------------------------------------------------------
   *  Test: Cut Robot from a part of sensor data
   * -------------------------------------------------------------------------------------------------------------
   */
  printf("\n\n Test %u:\n", test_results.size()+1);
  printf("\n--- Testing exclusion of robot from sensor data (w/o RayCasting)---\n");

  printf("Inserting a small robot, where sensor data will only partly be identical to robot.\n");
  robot_map->insertBoxByIndices(Vector3ui(2, 2, 2), Vector3ui(6, 5, 5),
                                Voxel::eC_EXECUTION, Voxel::eVT_OCCUPIED);

  for (uint32_t y=0; y<sensor.data_height; y++)
  {
    for (uint32_t x=0; x<sensor.data_width; x++)
    {
      data_index = y*data_width + x;
      sensor_data[data_index] = Vector3f(x * voxel_side_length - (float)(data_width -1) * 0.5f * voxel_side_length,
                                         y * voxel_side_length - (float)(data_height-1) * 0.5f * voxel_side_length,
                                         sensor_data_distance);

      printf("SensorData(%u) = (%f, %f, %f)\n",
             data_index, sensor_data[data_index].x, sensor_data[data_index].y, sensor_data[data_index].z);
    }
  }

  printf("Inserting sensor data, excluding where the robot is.\n");
  update_counter = env_map->getUpdateCounter();
  env_map->insertSensorData(sensor_data, false, true, robot_map->getDeviceDataPtr());
  env_map->increaseUpdateCounter();

  if (update_counter+1 == env_map->getUpdateCounter())
  {
    counter_updated = true;
  }
  else
  {
    counter_updated = false;
  }

  printf("Performing a collision check that should NOT result in a collision.\n");
  printf("Collision check...\n");
  collision = env_map->collisionCheck(col_check_threshold, robot_map, col_check_threshold, Voxel::eC_EXECUTION);
  printf("There was %s collision.\n", (collision? "a" : "no"));

  test_results.push_back((!collision && counter_updated));

  printf("Press key to proceed to next test.\n");
  std::cin >> key;

  printf("Clearing voxelmaps.. ");
  env_map->clearVoxelMap(Voxel::eC_EXECUTION);
  robot_map->clearVoxelMap(Voxel::eC_EXECUTION);
  printf(" .. done\n");


  /* -------------------------------------------------------------------------------------------------------------
   *  Test: Cut Robot where whole sensor data is within robot range, now with RayCasting
   * -------------------------------------------------------------------------------------------------------------
   */
  printf("\n\n Test %u:\n", test_results.size()+1);
  printf("\n--- Testing exclusion of robot from sensor data (with RayCasting)---\n");

  printf("Inserting a robot.\n");
  robot_map->insertBoxByIndices(Vector3ui(3, 3, 3), Vector3ui(8, 8, 8),
                                  Voxel::eC_EXECUTION, Voxel::eVT_OCCUPIED);

  for (uint32_t y=0; y<sensor.data_height; y++)
  {
    for (uint32_t x=0; x<sensor.data_width; x++)
    {
      data_index = y*data_width + x;
      sensor_data[data_index] = Vector3f(x * voxel_side_length - (float)(data_width -1) * 0.5f * voxel_side_length,
                                         y * voxel_side_length - (float)(data_height-1) * 0.5f * voxel_side_length,
                                         sensor_data_distance);

      printf("SensorData(%u) = (%f, %f, %f)\n",
             data_index, sensor_data[data_index].x, sensor_data[data_index].y, sensor_data[data_index].z);
    }
  }

  printf("Inserting sensor data, excluding where the robot is.\n");
  update_counter = env_map->getUpdateCounter();
  env_map->insertSensorData(sensor_data, true, true, robot_map->getDeviceDataPtr());
  env_map->increaseUpdateCounter();

  if (update_counter+1 == env_map->getUpdateCounter())
  {
    counter_updated = true;
  }
  else
  {
    counter_updated = false;
  }

  printf("Performing a collision check that should NOT result in a collision.\n");
  printf("Collision check...\n");
  collision = env_map->collisionCheck(col_check_threshold, robot_map, col_check_threshold, Voxel::eC_EXECUTION);
  printf("There was %s collision.\n", (collision? "a" : "no"));

  test_results.push_back((!collision && counter_updated));

  printf("Press key to proceed to next test.\n");
  std::cin >> key;


  printf("Clearing voxelmaps.. ");
  env_map->clearVoxelMap(Voxel::eC_EXECUTION);
  robot_map->clearVoxelMap(Voxel::eC_EXECUTION);
  printf(" .. done\n");

  /* -------------------------------------------------------------------------------------------------------------
   *  Test: Cut Robot from a part of sensor data
   * -------------------------------------------------------------------------------------------------------------
   */
  printf("\n\n Test %u:\n", test_results.size()+1);
  printf("\n--- Testing exclusion of robot from sensor data (with RayCasting)---\n");

  printf("Inserting a small robot, where sensor data will only partly be identical to robot.\n");
  robot_map->insertBoxByIndices(Vector3ui(2, 2, 2), Vector3ui(6, 5, 5),
                                Voxel::eC_EXECUTION, Voxel::eVT_OCCUPIED);

  for (uint32_t y=0; y<sensor.data_height; y++)
  {
    for (uint32_t x=0; x<sensor.data_width; x++)
    {
      data_index = y*data_width + x;
      sensor_data[data_index] = Vector3f(x * voxel_side_length - (float)(data_width -1) * 0.5f * voxel_side_length,
                                         y * voxel_side_length - (float)(data_height-1) * 0.5f * voxel_side_length,
                                         sensor_data_distance);

      printf("SensorData(%u) = (%f, %f, %f)\n",
             data_index, sensor_data[data_index].x, sensor_data[data_index].y, sensor_data[data_index].z);
    }
  }

  printf("Inserting sensor data, excluding where the robot is.\n");
  update_counter = env_map->getUpdateCounter();
  env_map->insertSensorData(sensor_data, true, true, robot_map->getDeviceDataPtr());
  env_map->increaseUpdateCounter();

  if (update_counter+1 == env_map->getUpdateCounter())
  {
    counter_updated = true;
  }
  else
  {
    counter_updated = false;
  }

  printf("Performing a collision check that should NOT result in a collision.\n");
  printf("Collision check...\n");
  collision = env_map->collisionCheck(col_check_threshold, robot_map, col_check_threshold, Voxel::eC_EXECUTION);
  printf("There was %s collision.\n", (collision? "a" : "no"));

  test_results.push_back((!collision && counter_updated));

  printf("Press key to proceed to next test.\n");
  std::cin >> key;

  printf("Clearing voxelmaps.. ");
  env_map->clearVoxelMap(Voxel::eC_EXECUTION);
  robot_map->clearVoxelMap(Voxel::eC_EXECUTION);
  printf(" .. done\n");

  // ================== END OF TESTS ==================

  printf("\n\nSummary:\n");
  printf("========\n\n");
  for (uint32_t test_nr=0; test_nr < test_results.size(); test_nr++)
  {
    printf(" Test %u: %s\n\n", test_nr+1, (test_results[test_nr] ? "passed" : "FAILED"));
  }

  vis_thread.join();
  delete sensor_data;
  delete env_map;
  delete robot_map;
  printf("Exiting..\n");
  return 0;
}
