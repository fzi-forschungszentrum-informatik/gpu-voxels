#ifndef IBEONCOMFILTER_H
#define IBEONCOMFILTER_H

#include <icl_sourcesink/DataSinkRegistration.h>
#include <icl_sourcesink/DataFilter.h>
#include <icl_sourcesink_example_components/ImportExport.h>
#include <icl_hardware_ibeo_noapi/DataTypes.h>
#include <icl_sourcesink/SourceSinkManager.h>
#include <icl_gps_types/NEDPose.h>
#include <gpu_voxels/GpuVoxels.h>
#include <icl_hardware_ncom/NcomSource.h>
#include <icl_velodyne/VelodyneTypes.h>
#include <icl_velodyne/VelodynePCLPointCloud.h>
//#include <gpu_voxels/helpers/CudaMath.h>
#include <cuda_runtime.h>
#include <gpu_classification/filter/IbeoNcomFilterGPU.h>
#include <gpu_classification/SegmentationHelper.h>

namespace gpu_voxels{
namespace classification{

using namespace gpu_voxels;
using namespace icl_hardware::ibeo;

class ICL_SOURCESINK_EXAMPLES_IMPORT_EXPORT IbeoNcomFilter : public icl_sourcesink::DataFilter<IbeoMsg, std::vector<gpu_voxels::Vector3f> >
{
public:

  /* Typedef the pointer.
   * Otherwise a generic DataSink::Ptr will be
   * used when calling UInt32ExponentialFilter::Ptr
   * which generally may not be intended.
   */
  typedef boost::shared_ptr<IbeoNcomFilter> Ptr;

  // Here we define all URI schemes the filter supports
  static icl_sourcesink::URISchemeMap supportedURISchemes()
  {
    using namespace icl_sourcesink;
    URISchemeMap schemes;
    schemes.insert(
          std::make_pair(
            "ibeo-metapointcloud+ncom",
            URISchemeInfo(
              "Filters Ibeo with Ncom-data to a transformed pointlcoud.",
              "ibeo-metapointcloud+ncom:",
              "no parameters needed \n")));
    return schemes;
  }

  //! Constructor
  IbeoNcomFilter(const std::string& uri = "IbeoNcomFilter",
                 const std::string& name = "UnnamedIbeoNcomFilter");

  //! Destructor
  ~IbeoNcomFilter();

  /* The set() function is called to set new data
   */
  virtual void set(const icl_core::Stamped<IbeoMsg>::Ptr& data);

  /* set the NcomData before you call the set() method
   */
  void setNcomData(const icl_core::Stamped<icl_hardware::ncom::Ncom>::Ptr& data);

  /*
   */
  void setVelodynePointCloudData(const icl_core::Stamped<VelodynePCLPointCloud::PointCloud>::Ptr& data);
  //  void setVelodynePointCloudDataStamped(const icl_core::Stamped<VelodynePCLPointCloud::PointCloud> data);

  void setVelodyneRotationAngle(float angle);

  void setFilterParameter(bool displayGround, bool displayObjects, bool displayReference, bool displayClasses, bool classificationToggle, Vector3f add_trans);

  void setGroundSegmentationParameters(double segmentAngle, int numberOfBinsPerSegment, float rangeMin, float rangeMax, float thresholdLineGradient, float thresholdHorizontalLineGradient, float thresholdLowestGroundHeight, float thresholdDistancePointToLine);

  void setObjectSegmentationParameters(float relativeDistanceThreshold, float absoluteDistanceThreshold, float levelOfConcavity, float levelOfNoise, float zSimilarity, float aabbScaling, float normalMerge);

  void setClassificationClasses(std::vector<ClassificationClass> classes);

  std::vector<ClassificationClass> getClassificationClasses();

  void processPointcloud();

  void redrawPointcloud();
  /* The filter() function does the actual filtering. It can be run on its own
   * or by the set function.
   */
  virtual bool filter(const icl_core::Stamped<IbeoMsg>& input_data, icl_core::Stamped<std::vector<gpu_voxels::Vector3f> >& output_data);

  /* The get() function is called when a user wants to
   * access the data currently availabe
   */
  virtual icl_core::Stamped<std::vector<gpu_voxels::Vector3f> >::Ptr get() const;

private:

  std::string pointCloudName;
  uint currentNumberOfPointclouds;
  uint iterationCount;

  bool displayGround;
  bool displayObjects;
  bool displayReference;
  bool displayClasses;

  icl_core::Stamped<std::vector<gpu_voxels::Vector3f> >::Ptr m_data;

  icl_gps::NEDPose* ref_pose;

  icl_hardware::ncom::NcomStamped::Ptr ncom_element;
  icl_core::Stamped<VelodynePCLPointCloud::PointCloud>::Ptr velodyne_element;
  icl_core::Stamped<VelodynePCLPointCloud::PointCloud> velodyne_stamped;

  gpu_voxels::MetaPointCloud mpc;
  gpu_voxels::MetaPointCloud* transformMPC;
  int16_t cloud_number;

  float velodyneRotationAngle;

  Matrix4f transform_matrix;
  Matrix4f* transform_matrix_dev;
  uint32_t m_blocks;
  uint32_t m_threads_per_block;
  gpu_voxels::Vector3f m_additional_translation;

  IbeoNcomFilterGPU gpu;

  boost::shared_ptr<gpu_voxels::GpuVoxels> gvl;

  std::vector<gpu_voxels::BitVoxelMeaning> objectMeanings;

  std::vector<Vector3f> ibeoCloud;
  std::vector<Vector3f> velodyneCloud;
  std::vector<Vector3f> point_cloud;


  //TEST Objects
  SegmentationHelper sh;

  //void transformPointCloud();

  void visualizePointClouds(MetaPointCloud* mpc, std::vector<gpu_voxels::BitVoxelMeaning> bitMeanings);
  void visualizePointCloud(MetaPointCloud* mpc);
  void testTransformPointcloud();
};

SOURCESINK_DECLARE_GENERIC_SINK_FACTORY(IbeoNcomFilter)

}
}

#endif // IBEONCOMFILTER_H
