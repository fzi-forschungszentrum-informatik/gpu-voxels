
#include <cstdlib>
#include <signal.h>
#include <typeinfo>

#include <icl_core_config/Config.h>

#include <gpu_voxels/GpuVoxels.h>
#include <gpu_voxels/helpers/PointCloud.h>
#include <gpu_voxels/octree/Octree.h>
#include <gpu_voxels/logging/logging_gpu_voxels.h>

using namespace gpu_voxels;


void ctrlchandler(int)
{
  //delete gvl;
  exit(EXIT_SUCCESS);
}
void killhandler(int)
{
  //delete gvl;
  exit(EXIT_SUCCESS);
}

int main(int argc, char* argv[])
{
  signal(SIGINT, ctrlchandler);
  signal(SIGTERM, killhandler);

  icl_core::logging::initialize(argc, argv);

  //Testing Singleton of GpuVoxels
  LOGGING_INFO(Gpu_voxels, "Start Singleton Test" << endl);
  GpuVoxelsSharedPtr gvl1 = GpuVoxels::getInstance();
  uint32_t gvlX, gvlY, gvlZ;
  float gvlSide;
  gvl1->getDimensions(gvlX, gvlY, gvlZ);
  gvl1->getVoxelSideLength(gvlSide);
  LOGGING_INFO(Gpu_voxels, "Gvl 1Pre: (" << gvlX << ", " << gvlY << ", " << gvlZ << ") Sidelength: " << gvlSide << " Pointer: " << gvl1 << endl);
  gvl1->initialize(200, 200, 100, 0.01);
  LOGGING_INFO(Gpu_voxels, "GVL instance count: " << gvl1.use_count() << endl);
  GpuVoxelsSharedPtr gvl2 = GpuVoxels::getInstance();
  gvl2->initialize(400, 400, 200, 0.01);
  LOGGING_INFO(Gpu_voxels, "GVL instance count: " << gvl2.use_count() << endl);

  gvl1->getDimensions(gvlX, gvlY, gvlZ);
  gvl1->getVoxelSideLength(gvlSide);
  LOGGING_INFO(Gpu_voxels, "Gvl 1Pre: (" << gvlX << ", " << gvlY << ", " << gvlZ << ") Sidelength: " << gvlSide << " Pointer: " << gvl1 << endl);
  gvl2->getDimensions(gvlX, gvlY, gvlZ);
  gvl2->getVoxelSideLength(gvlSide);
  LOGGING_INFO(Gpu_voxels, "Gvl 1Pre: (" << gvlX << ", " << gvlY << ", " << gvlZ << ") Sidelength: " << gvlSide << " Pointer: " << gvl2 << endl);

  gvl2.reset();
  LOGGING_INFO(Gpu_voxels, "GVL instance count: " << gvl1.use_count() << endl);
  gvl1.reset();
  LOGGING_INFO(Gpu_voxels, "GVL instance count: " << gvl1.use_count() << endl);
  GpuVoxelsSharedPtr gvl3 = GpuVoxels::getInstance();
  gvl3->initialize(300, 300, 100, 0.01);
  LOGGING_INFO(Gpu_voxels, "Created GVL3. GVL instance count: " << gvl3.use_count() << endl);
  gvl3->getDimensions(gvlX, gvlY, gvlZ);
  gvl3->getVoxelSideLength(gvlSide);
  LOGGING_INFO(Gpu_voxels, "Gvl 1Pre: (" << gvlX << ", " << gvlY << ", " << gvlZ << ") Sidelength: " << gvlSide << " Pointer: " << gvl3 << endl);
  gvl3.reset();


  //Testing as-Operator of class GpuVoxelsMap
  LOGGING_INFO(Gpu_voxels, "Start Maps Conversion Test" << endl);
  GpuVoxelsMapSharedPtr bitVoxelListMap1(new voxellist::BitVectorVoxelList(Vector3ui(100,100,100), 0.01f, MT_BITVECTOR_VOXELLIST));
  GpuVoxelsMapSharedPtr bitVoxelListMap2(new voxellist::BitVectorVoxelList(Vector3ui(200,200,200), 0.02f, MT_BITVECTOR_VOXELLIST));


  GpuVoxelsMapSharedPtr probVoxelMapMap1(new voxelmap::ProbVoxelMap(100, 100, 100, 0.01f, MT_PROBAB_VOXELMAP));
  GpuVoxelsMapSharedPtr probVoxelMapMap2(new voxelmap::ProbVoxelMap(200, 200, 200, 0.02f, MT_PROBAB_VOXELMAP));


  GpuVoxelsMapSharedPtr bitVoxelMapMap1(new voxelmap::BitVectorVoxelMap(100, 100, 100, 0.01f, MT_BITVECTOR_VOXELMAP));
  GpuVoxelsMapSharedPtr bitVoxelMapMap2(new voxelmap::BitVectorVoxelMap(200, 200, 200, 0.2f, MT_BITVECTOR_VOXELMAP));


  GpuVoxelsMapSharedPtr bitOctreeMap(new NTree::GvlNTreeDet(0.01f, MT_BITVECTOR_OCTREE));
  GpuVoxelsMapSharedPtr probOctreeMap(new NTree::GvlNTreeProb(0.1f, MT_PROBAB_OCTREE));


  std::vector<GpuVoxelsMapSharedPtr> maps(0);
  maps.push_back(bitVoxelListMap1);
  maps.push_back(bitVoxelListMap2);
  maps.push_back(probVoxelMapMap1);
  maps.push_back(probVoxelMapMap2);
  maps.push_back(bitVoxelMapMap1);
  maps.push_back(bitVoxelMapMap2);
  maps.push_back(bitOctreeMap);
  maps.push_back(probOctreeMap);

  LOGGING_INFO(Gpu_voxels, "start checking maps" << endl);

  for(uint i = 0; i < maps.size(); i++)
  {
    if(maps[i]->is<voxellist::BitVectorVoxelList>())
    {
      Vector3ui temp = maps[i]->as<voxellist::BitVectorVoxelList>()->getDimensions();
      LOGGING_INFO(Gpu_voxels, "I am a BitVectorVoxelList: Dimensions: " << temp << endl);
    }
    else if(maps[i]->is<voxelmap::ProbVoxelMap>())
    {
      Vector3ui temp = maps[i]->as<voxelmap::ProbVoxelMap>()->getDimensions();
      LOGGING_INFO(Gpu_voxels, "I am a ProbVoxelMap: Dimensions: " << temp << endl);
    }
    else if(maps[i]->is<voxelmap::BitVectorVoxelMap>())
    {
      Vector3ui temp = maps[i]->as<voxelmap::BitVectorVoxelMap>()->getDimensions();
      LOGGING_INFO(Gpu_voxels, "I am a BitVectorVoxelMap: Dimensions: " << temp << endl);
    }
    else if(maps[i]->is<NTree::GvlNTreeDet>())
    {
      Vector3ui temp = maps[i]->as<NTree::GvlNTreeDet>()->getDimensions();
      LOGGING_INFO(Gpu_voxels, "I am a GvlNTreeDet: Dimensions: " << temp << endl);
    }
    else if(maps[i]->is<NTree::GvlNTreeProb>())
    {
      Vector3ui temp = maps[i]->as<NTree::GvlNTreeProb>()->getDimensions();
      LOGGING_INFO(Gpu_voxels, "I am a GvlNTreeProb: Dimensions: " << temp << endl);
    }
    else
    {
      LOGGING_INFO(Gpu_voxels, "I am an UNKNOWN Map" << endl);
    }
  }

  sleep(1);
  //Testing PointCloud implementation

  std::vector<gpu_voxels::Vector3f> vectorCloud;
  vectorCloud.push_back(gpu_voxels::Vector3f(0.0f, 0.0f, 0.0f));
  vectorCloud.push_back(gpu_voxels::Vector3f(1.0f, 0.0f, 0.0f));
  vectorCloud.push_back(gpu_voxels::Vector3f(0.0f, 1.0f, 0.0f));
  vectorCloud.push_back(gpu_voxels::Vector3f(0.0f, 0.0f, 1.0f));

  gpu_voxels::PointCloud cloud1(vectorCloud);
  cloud1.print();

  gpu_voxels::Matrix4f transform;
  transform.a11 = transform.a22 = transform.a33 = transform.a44 = 1.0f;
  transform.a14 = 5.0f;
  transform.a24 = 5.0f;
  transform.a34 = 5.0f;
  transform.print();

  cloud1.transformSelf(&transform);
  cloud1.print();

  gpu_voxels::Vector3f pointerCloud [4] = {gpu_voxels::Vector3f(2.0f, 0.0f, 0.0f),
                                           gpu_voxels::Vector3f(0.0f, 2.0f, 0.0f),
                                           gpu_voxels::Vector3f(0.0f, 0.0f, 2.0f),
                                           gpu_voxels::Vector3f(2.0f, 2.0f, 2.0f)};

  gpu_voxels::PointCloud cloud2(&pointerCloud[0], 4);

  cloud1.transform(&transform, &cloud2);
  cloud1.print();
  cloud2.print();

  cloud2.add(&cloud1);
  cloud2.add(vectorCloud);
  cloud2.add(&pointerCloud[0], 3);
  std::vector<gpu_voxels::Vector3f> tmp;
  tmp.push_back(gpu_voxels::Vector3f(3.0f, 3.0f, 3.0f));
  cloud2.add(tmp);

  cloud2.print();

  cloud2.update(&pointerCloud[0], 3);
  cloud2.print();

  cloud2.update(&cloud1);
  cloud2.print();

  cloud2.update(vectorCloud);
  cloud2.print();
}
