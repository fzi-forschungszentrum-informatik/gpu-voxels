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

#ifndef IBEOSINKGPUVOXEL_H
#define IBEOSINKGPUVOXEL_H


#include <icl_sourcesink/DataSinkRegistration.h>
#include <icl_hardware_ibeo_noapi/sink/IbeoSinkNoAPI.h>
#include <gpu_voxels/GpuVoxels.h>

namespace gpu_voxels {
namespace classification {

using namespace icl_hardware::ibeo;

class ICL_HARDWARE_IBEO_NOAPI_IMPORT_EXPORT IbeoSinkGPUVoxel : public IbeoSinkNoAPI
{
public:
  //! Shared pointer shorthand.
  typedef boost::shared_ptr<IbeoSinkGPUVoxel> Ptr;
  //! Const shared pointer shorthand.
  typedef boost::shared_ptr<const IbeoSinkGPUVoxel> ConstPtr;

  static icl_sourcesink::URISchemeMap supportedURISchemes()
  {
    using namespace icl_sourcesink;
    URISchemeMap schemes;
    schemes.insert(
          std::make_pair(
            "ibeo+gpuvoxel",
            URISchemeInfo(
              "Writes IbeoMsg data to the GPU-Voxel-Visualizer.",
              "ibeo+gpuvoxel:<toClear>",
              "<toClear> type \"clear\" if the map should be cleared with each new pointcloud")));
    return schemes;
  }


  //! Constructor
  IbeoSinkGPUVoxel(const std::string& uri = "IbeoSinkGPUVoxel",
                   const std::string& name = "UnnamedIbeoSinkGPUVoxel");


  virtual void set(const typename icl_core::Stamped<IbeoMsg>::Ptr& data);

private:

  boost::shared_ptr<gpu_voxels::GpuVoxels> gvl;
};


SOURCESINK_DECLARE_GENERIC_SINK_FACTORY(IbeoSinkGPUVoxel)
}
}
#endif // IBEOSINKGPUVOXEL_H
