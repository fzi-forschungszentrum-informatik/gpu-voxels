#ifndef METAPOINTCLOUDSINKGPUVOXEL_H
#define METAPOINTCLOUDSINKGPUVOXEL_H

#include <icl_sourcesink/DataSink.h>
#include <gpu_voxels/GpuVoxels.h>
#include <icl_sourcesink/DataSinkRegistration.h>
#include <icl_sourcesink_example_components/ImportExport.h>

namespace gpu_voxels{
namespace classification{

using namespace gpu_voxels;
using namespace icl_sourcesink;

class ICL_SOURCESINK_EXAMPLES_IMPORT_EXPORT MetapointcloudSinkGpuvoxel : public DataSink<MetaPointCloud>
{
public:
  //! Shared pointer shorthand.
  typedef boost::shared_ptr<MetapointcloudSinkGpuvoxel> Ptr;
  //! Const shared pointer shorthand.
  typedef boost::shared_ptr<const MetapointcloudSinkGpuvoxel> ConstPtr;

  static icl_sourcesink::URISchemeMap supportedURISchemes()
  {
    using namespace icl_sourcesink;
    URISchemeMap schemes;
    schemes.insert(
          std::make_pair(
            "metapointcloud+gpuvoxel",
            URISchemeInfo(
              "Writes metapointcloud to the GPU-Voxel-Visualizer.",
              "pointcloud+gpuvoxel:<toClear>",
              "<toClear> type \"clear\" if the map should be cleared with each new pointcloud")));
    return schemes;
  }


  MetapointcloudSinkGpuvoxel(const std::string& uri = "MetapointcloudSinkGpuvoxel",
                             const std::string& name = "UnnamedMetapointcloudSinkGpuvoxel");


  virtual void set(const typename icl_core::Stamped<MetaPointCloud>::Ptr& data);

private:

  boost::shared_ptr<gpu_voxels::GpuVoxels> gvl;
};

SOURCESINK_DECLARE_GENERIC_SINK_FACTORY(MetapointcloudSinkGpuvoxel)

}
}
#endif // METAPOINTCLOUDSINKGPUVOXEL_H
