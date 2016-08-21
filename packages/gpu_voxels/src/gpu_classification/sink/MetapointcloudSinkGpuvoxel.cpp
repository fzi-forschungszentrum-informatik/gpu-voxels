#include "MetapointcloudSinkGpuvoxel.h"

#include <gpu_voxels/logging/logging_gpu_voxels.h>
#include "icl_sourcesink/SimpleURI.h"


namespace gpu_voxels {
namespace classification {

using namespace gpu_voxels;

MetapointcloudSinkGpuvoxel::MetapointcloudSinkGpuvoxel(const std::string& uri,
                                                       const std::string& name)
  : DataSink<MetaPointCloud>(uri, name)
{
  //create new GpuVoxels Object of defined size
  gvl = GpuVoxels::getInstance();
  gvl->initialize(200, 200, 200, 0.1);


  //create new Map
  gvl->addMap(MT_BITVECTOR_OCTREE, "myPointcloudMap");
}

void MetapointcloudSinkGpuvoxel::set(const icl_core::Stamped<MetaPointCloud>::Ptr &data)
{
  icl_sourcesink::SimpleURI parsed_uri(uri());
  std::string param = parsed_uri.path();

  if(!param.compare("clear"))
  {
    gvl->clearMap("myPointcloudMap");
  }

  //    std::cout << "\nafter clear\n";
  //insert pointcloud into Map

  MetaPointCloud point_cloud = *data;

  //    std::cout << "POINTcloud:" << point_cloud.size() << std::endl;
  //      std::cout << "\n\n METAPOINTCLOUD: " << point_cloud.getNumberOfPointclouds() << std::endl;
  //    if(point_cloud.getNumberOfPointclouds() > 0)
  //    {
  std::cout << "METAPOINTCLOUD: " << point_cloud.getNumberOfPointclouds() << " anzahl: " << point_cloud.getAccumulatedPointcloudSize() << std::endl;

  //gvl->getMap("myPointcloudMap")->insertMetaPointCloud(point_cloud, eVT_OCCUPIED);
  //    }
  //    std::cout << "\ninserted in the map\n";
  //gvl->visualizeMap("myPointcloudMap");
  //    std::cout << "\nvisualized the map\n";
}

}
}
