#include "PointcloudSinkGpuvoxel.h"

#include <gpu_voxels/logging/logging_gpu_voxels.h>


namespace gpu_voxels {
namespace classification {

using namespace gpu_voxels;


PointcloudSinkGpuvoxel::PointcloudSinkGpuvoxel(const std::string& uri,
                                               const std::string& name)
  : DataSink<std::vector<Vector3f> >(uri, name)
{
  //create new GpuVoxels Object of defined size
  gvl = GpuVoxels::getInstance();
  gvl->initialize(200, 200, 200, 0.1);


  //create new Map
  gvl->addMap(MT_BITVECTOR_OCTREE, "myPointcloudMap");
}

void PointcloudSinkGpuvoxel::set(const icl_core::Stamped<std::vector<Vector3f> >::Ptr &data)
{
  gvl->clearMap("myPointcloudMap");

  //insert pointcloud into Map

  std::vector<Vector3f> point_cloud = *data;

  //    std::cout << "POINTcloud:" << point_cloud.size() << std::endl;

  if(point_cloud.size() > 0)
  {
    gvl->getMap("myPointcloudMap")->insertPointCloud(point_cloud, eBVM_OCCUPIED);
  }

  gvl->visualizeMap("myPointcloudMap");

}

}
}
