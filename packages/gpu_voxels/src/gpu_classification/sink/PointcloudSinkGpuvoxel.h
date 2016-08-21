#ifndef POINTCLOUDSINKSPUVOXEL_H
#define POINTCLOUDSINKSPUVOXEL_H

#include <icl_sourcesink/DataSink.h>
#include <gpu_voxels/GpuVoxels.h>
#include <icl_sourcesink/DataSinkRegistration.h>
#include <icl_sourcesink_example_components/ImportExport.h>

namespace gpu_voxels{
namespace classification{

using namespace gpu_voxels;
using namespace icl_sourcesink;

class ICL_SOURCESINK_EXAMPLES_IMPORT_EXPORT PointcloudSinkGpuvoxel : public DataSink<std::vector<Vector3f> >
{
public:
  //! Shared pointer shorthand.
  typedef boost::shared_ptr<PointcloudSinkGpuvoxel> Ptr;
  //! Const shared pointer shorthand.
  typedef boost::shared_ptr<const PointcloudSinkGpuvoxel> ConstPtr;

  static icl_sourcesink::URISchemeMap supportedURISchemes()
  {
    using namespace icl_sourcesink;
    URISchemeMap schemes;
    schemes.insert(
          std::make_pair(
            "pointcloud+gpuvoxel",
            URISchemeInfo(
              "Writes PCL pointcloud to the GPU-Voxel-Visualizer.",
              "pointcloud+gpuvoxel:",
              "no parameters needed")));
    return schemes;
  }


  PointcloudSinkGpuvoxel(const std::string& uri = "PointcloudSinkGpuvoxel",
                         const std::string& name = "UnnamedPointcloudSinkGpuvoxel");


  virtual void set(const typename icl_core::Stamped<std::vector<Vector3f> >::Ptr& data);

private:

  boost::shared_ptr<gpu_voxels::GpuVoxels> gvl;
};

SOURCESINK_DECLARE_GENERIC_SINK_FACTORY(PointcloudSinkGpuvoxel)

}
}
#endif // POINTCLOUDSINKSPUVOXEL_H
