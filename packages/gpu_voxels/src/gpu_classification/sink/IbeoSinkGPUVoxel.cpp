// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 *\author   Herbert Pietrzyk
 *\date     2015-07-28
 *
 */
//----------------------------------------------------------------------

#include "IbeoSinkGPUVoxel.h"

#include <gpu_voxels/logging/logging_gpu_voxels.h>
#include "icl_sourcesink/SimpleURI.h"

namespace gpu_voxels {
namespace classification {

using namespace gpu_voxels;
namespace nibeo = icl_hardware::ibeo;

IbeoSinkGPUVoxel::IbeoSinkGPUVoxel(const std::string& uri,
                                   const std::string& name)
  : IbeoSinkNoAPI(uri, name)
{
  //create new GpuVoxels Object of defined size
  gvl = GpuVoxels::getInstance();
  gvl->initialize(200, 200, 200, 0.1);


  //create new Map
  gvl->addMap(MT_BITVECTOR_OCTREE, "myIbeoMap");
}



void IbeoSinkGPUVoxel::set(const icl_core::Stamped<IbeoMsg>::Ptr &data)
{

  icl_sourcesink::SimpleURI parsed_uri(uri());
  std::string param = parsed_uri.path();

  if(!param.compare("clear"))
  {
    gvl->clearMap("myIbeoMap");
  }


  //only use Scan Data from Ibeo
  nibeo::IbeoScanMsg scan_msg;
  nibeo::IbeoObjectMsg obj_msg;
  if(!scan_msg.fromIbeoMsg(*data))
  {return;}
  if(!obj_msg.fromIbeoMsg(*data))
  {    }

  LOGGING_INFO(Gpu_voxels, "Number of points in ibeo msg: " << scan_msg.number_of_points
               << " Number of Objects: " << obj_msg.number_of_objects << endl);

  //convert to Vector
  std::vector<Vector3f> point_cloud(scan_msg.number_of_points);
  for(int i = 0; i < scan_msg.number_of_points; i++)
  {
    point_cloud[i].x = (*scan_msg.scan_points)[i].x;
    point_cloud[i].y = (*scan_msg.scan_points)[i].y;
    point_cloud[i].z = (*scan_msg.scan_points)[i].z;
  }

  //insert pointcloud from Ibeo into Map
  gvl->getMap("myIbeoMap")->insertPointCloud(point_cloud, eBVM_OCCUPIED);

  gvl->visualizeMap("myIbeoMap");

}


}
}
