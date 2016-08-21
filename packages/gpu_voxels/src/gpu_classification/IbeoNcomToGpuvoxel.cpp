#include <icl_core/os.h>
#include <icl_core_logging/Logging.h>
#include <icl_core_config/Config.h>
#include <icl_velodyne/VelodyneTypes.h>
#include <gpu_classification/logging/logging_classification.h>

#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>

//sources
#include <icl_hardware_ibeo_noapi/source/IbeoSources.h>
#include <icl_hardware_ncom/NcomSources.h>
#include <icl_velodyne/source/VelodyneSources.h>


//filter
#include <gpu_classification/filter/IbeoNcomFilter.h>
#include <icl_velodyne/filter/VelodyneToPointCloudFilter.h>

//sink
#include <gpu_classification/sink/PointcloudSinkGpuvoxel.h>

//#include <gpu_voxels/robot/kernels/KinematicOperations.h>
#include <gpu_classification/logging/logging_classification.h>

#include "classificator_reconfigure/ClassificatorConfig.h"


#include <cstdlib>
#include <signal.h>
#include <cstdio>

namespace icss = icl_sourcesink;
namespace nncom = icl_hardware::ncom;
namespace nibeo = icl_hardware::ibeo;
namespace nvelo = velodyne;
namespace gpuclass = gpu_voxels::classification;

gpuclass::IbeoNcomFilter::Ptr filter;
std::vector<gpuclass::ClassificationClass> classes;
bool stopPreview;

void ctrlchandler(int)
{
  ros::shutdown();
  exit(EXIT_SUCCESS);
}
void killhandler(int)
{
  ros::shutdown();
  exit(EXIT_SUCCESS);
}


void callback(classificator_reconfigure::ClassificatorConfig &config, uint32_t level) {
  /*ROS_INFO("Reconfigure Request: %d %f %s %s %d",
            config.int_param, config.double_param,
            config.str_param.c_str(),
            config.bool_param?"True":"False",
            config.size);*/

  ROS_INFO("Reconfigure Callback: %s, %d", config.className.c_str(), config.meaning);
  gpuclass::ClassificationClass rosClass(config.className, config.meaning, config.parallelToGroundMax, config.parallelToGroundMin , config.centerHeightAboveGround, config.distanceOfCenters,
                                         config.centerParallelToGround, config.centerUp, config.areaMin, config.areaMax,
                                         config.volumeMin, config.volumeMax, config.shouldBeTall, config.verticalRatio,
                                         config.shouldBeQuadratic, config.horizontalRatio, config.densityMin, config. densityMax);

  stopPreview = config.stopPreview;

  if(classes.size() > 0)
  {
    classes[0].setParameters(config.className, config.meaning, config.parallelToGroundMax, config.parallelToGroundMin, config.centerHeightAboveGround, config.distanceOfCenters,
                             config.centerParallelToGround, config.centerUp, config.areaMin, config.areaMax,
                             config.volumeMin, config.volumeMax, config.shouldBeTall, config.verticalRatio,
                             config.shouldBeQuadratic, config.horizontalRatio, config.densityMin, config. densityMax);
  }
  else
  {
    classes.push_back(rosClass);
  }

  if(filter)
  {
    filter->setFilterParameter(config.displayGround, config.displayObjects, config.displayReference, config. displayClasses, config.classificationToggle,
                               gpu_voxels::Vector3f(config.additionalTransX, config.additionalTransY, config.additionalTransZ));
    filter->setGroundSegmentationParameters(config.segmentAngle, config.numberOfBins, config.rangeMin, config.rangeMax,
                                            config.thresholdLineGradient, config.thresholdHorizontalLineGradient,
                                            config.thresholdLowestGroundHeight, config.thresholdDistancePointToLine);
    filter->setObjectSegmentationParameters(config.relativeDistanceThreshold, config.absoluteDistanceThreshold,
                                            config.levelOfConcavity, config.levelOfNoise, config.zSimilarity,
                                            config.aabbScaling, config.normalMerge);
    filter->setClassificationClasses(classes);
    filter->setVelodyneRotationAngle(config.velodyneRotationAngle);

  }
}


int main(int argc, char *argv[])
{
  signal(SIGINT, ctrlchandler);
  signal(SIGTERM, killhandler);

  ros::init(argc, argv, "classificator_node", ros::init_options::NoSigintHandler);

  dynamic_reconfigure::Server<classificator_reconfigure::ClassificatorConfig> server;
  dynamic_reconfigure::Server<classificator_reconfigure::ClassificatorConfig>::CallbackType f;

  f = boost::bind(&callback, _1, _2);
  server.setCallback(f);

  ros::AsyncSpinner spinner(1);
  spinner.start();

  /*icl_core::config::GetoptParameter segmentAngle_param("segmentAngle:", "sA", "angle one Segment has in the groundsegmentation algorithm");
    icl_core::config::Getopt::instance().addParameter(segmentAngle_param);
    icl_core::config::GetoptParameter numberOfBins_param("numberOfBins:", "noB", "number of bins one segment has in the groundsegmentation algorithm");
    icl_core::config::Getopt::instance().addParameter(numberOfBins_param);
    icl_core::config::GetoptParameter rangeMin_param("rangeMin:", "rMin", "minimum range the groundsegmentation algorithm covers");
    icl_core::config::Getopt::instance().addParameter(rangeMin_param);
    icl_core::config::GetoptParameter rangeMax_param("rangeMax:", "rMax", "maximum range the groundsegmentation algorithm covers");
    icl_core::config::Getopt::instance().addParameter(rangeMax_param);
    icl_core::config::GetoptParameter thresholdLineGradient_param("thresholdLineGradient:", "tLG", "maxmimum gradient the lines of the segments are allowed to have in the groundsegmentation algorithm");
    icl_core::config::Getopt::instance().addParameter(thresholdLineGradient_param);
    icl_core::config::GetoptParameter thresholdHorizontalLineGradient_param("thresholdHorizontalLineGradient:", "tHLG", "minimum gradient a line of the segments are allowed to have to be counted as horizontal in the groundsegmentation algorithm");
    icl_core::config::Getopt::instance().addParameter(thresholdHorizontalLineGradient_param);
    icl_core::config::GetoptParameter thresholdLowestGroundHeight_param("thresholdLowestGroundHeight:", "tLGH", "minimum height the ground can have if it is horizontal in the groundsegmentation algorithm");
    icl_core::config::Getopt::instance().addParameter(thresholdLowestGroundHeight_param);
    icl_core::config::GetoptParameter thresholdDistancePointToLine_param("thresholdDistancePointToLine:", "tDPL", "maximum distance a point is allowed to have from the corresponding line to be labeled as ground in the groundsegmentation algorithm");
    icl_core::config::Getopt::instance().addParameter(thresholdDistancePointToLine_param);

    icl_core::config::GetoptParameter relativeDistanceThreshold_param("relativeDistanceThreshold:", "rDt", "relative distance a point is allowed to have to its neighbours in the objectsegmentation algorithm");
    icl_core::config::Getopt::instance().addParameter(relativeDistanceThreshold_param);
    icl_core::config::GetoptParameter absoluteDistanceThreshold_param("absoluteDistanceThreshold:", "aDt", "absolute distance no edge in the neighbourhood graph should exceed in the objectsegmentation algorithm");
    icl_core::config::Getopt::instance().addParameter(absoluteDistanceThreshold_param);
    icl_core::config::GetoptParameter levelOfConcavity_param("levelOfConcavity:", "loC", "level of Concavity in the objectsegmentation algorithm");
    icl_core::config::Getopt::instance().addParameter(levelOfConcavity_param);
    icl_core::config::GetoptParameter levelOfNoise_param("levelOfNoise:", "loN", "level of Noise in the objectsegmentation algorithm");
    icl_core::config::Getopt::instance().addParameter(levelOfNoise_param);
    icl_core::config::GetoptParameter zSimilarity_param("zSimilarity:", "zS", "defines how big the difference can be between two z-Coordinates to be counted as similar");
    icl_core::config::Getopt::instance().addParameter(zSimilarity_param);*/



  icl_core::logging::initialize(argc, argv);

  //paths to ncom and ibeo files
  const std::string ibeoFile = "/disk/hb/Neu/2015-01-23_09-29-16_Campus_Ost/20150123-092847.idc?offset=-2:00:00";
  const std::string ncomFile = "/disk/hb/Neu/2015-01-23_09-29-16_Campus_Ost/150123_081433.ncom?offset=-0:00:16.2";
  const std::string velodyneFile = "/disk/hb/Neu/2015-01-23_09-29-16_Campus_Ost/Aufnahme-0.vdyne.xml";
  const std::string velodyneConfFile = "/disk/hb/Neu/2015-01-23_09-29-16_Campus_Ost/32db.xml";

  icss::SourceSinkManager manager;

  //Create the sources
  nibeo::IbeoSourceNoAPI::Ptr ibeo_source;
  nncom::NcomSource::Ptr ncom_source;
  nvelo::VelodyneSource::Ptr velodyne_source;

  try
  {
    velodyne_source = manager.createSource<nvelo::VelodyneTypes::PacketFrame>("velodyne+file:" + velodyneFile, "velodynesource");
    ibeo_source = manager.createSource<nibeo::IbeoMsg>("ibeo+file:" + ibeoFile, "ibeosource");
    ncom_source = manager.createSource<nncom::Ncom>("ncom+file:" + ncomFile, "ncomsource");
    manager.setMasterSource(ibeo_source, true);
  }
  catch(const std::exception& e)
  {
    std::cerr << "At least one source could not be created!" << std::endl
              << "  " << e.what() << std::endl;
    return 1;
  }


  //create the filter
  filter = gpuclass::IbeoNcomFilter::Ptr(new gpuclass::IbeoNcomFilter("ibeo-pointcloud+ncom:", "filter"));

  filter->setFilterParameter(icl_core::config::getDefault<bool>("/general/displayGround", true),
                             icl_core::config::getDefault<bool>("/general/displayObjects", true),
                             icl_core::config::getDefault<bool>("/general/displayReference", false),
                             icl_core::config::getDefault<bool>("/general/displayClasses", true),
                             icl_core::config::getDefault<bool>("/general/classificationToggle", true),
                             gpu_voxels::Vector3f(icl_core::config::getDefault<float>("/general/additionaltranslation/x", 50.0f),
                                                  icl_core::config::getDefault<float>("/general/additionaltranslation/y", 50.0f),
                                                  icl_core::config::getDefault<float>("/general/additionaltranslation/z", 0.0f)));
  filter->setGroundSegmentationParameters(icl_core::config::getDefault<double>("/groundsegmentation/segmentAngle", 10.0f),
                                          icl_core::config::getDefault<int>("/groundsegmentation/numberOfBins", 50),
                                          icl_core::config::getDefault<float>("/groundsegmentation/rangeMin", 3.5f),
                                          icl_core::config::getDefault<float>("/groundsegmentation/rangeMax", 50.0f),
                                          icl_core::config::getDefault<float>("/groundsegmentation/thresholdLineGradient", 0.005f),
                                          icl_core::config::getDefault<float>("/groundsegmentation/thresholdHorizontalLineGradient", 0.02f),
                                          icl_core::config::getDefault<float>("/groundsegmentation/thresholdLowestGroundHeight", 0.2f),
                                          icl_core::config::getDefault<float>("/groundsegmentation/thresholdDistancePointToLine", 0.6f));
  filter->setObjectSegmentationParameters(icl_core::config::getDefault<float>("/objectsegmentation/relativeDistanceThreshold", 0.5f),
                                          icl_core::config::getDefault<float>("/objectsegmentation/absoluteDistanceThreshold", 1.0f),
                                          icl_core::config::getDefault<float>("/objectsegmentation/levelOfConcavity", 0.1f),
                                          icl_core::config::getDefault<float>("/objectsegmentation/levelOfNoise", 0.08f),
                                          icl_core::config::getDefault<float>("/objectsegmentation/zSimilarity", 0.5f),
                                          icl_core::config::getDefault<float>("/objectsegmentation/aabbScaling", 1.0f),
                                          icl_core::config::getDefault<float>("/objectsegmentation/normalMerge", 0.8f));

  int numberOfClasses = icl_core::config::getDefault<int>("/classification/numberOfClasses", 1);

  std::ostringstream stringStream;
  for(int i = 0; i < numberOfClasses; i++)
  {
    stringStream << "/classification/class_" << i << "/name";
    std::string name = icl_core::config::getDefault<std::string>(stringStream.str(), "KeinName");
    stringStream.str("");
    stringStream << "/classification/class_" << i << "/meaning";
    int meaning = icl_core::config::getDefault<int>(stringStream.str(), 18);
    stringStream.str("");
    stringStream << "/classification/class_" << i << "/parallelToGroundMax";
    float parallelToGroundMax = icl_core::config::getDefault<float>(stringStream.str(), 0.0f);
    stringStream.str("");
    stringStream << "/classification/class_" << i << "/parallelToGroundMin";
    float parallelToGroundMin = icl_core::config::getDefault<float>(stringStream.str(), 0.0f);
    stringStream.str("");
    stringStream << "/classification/class_" << i << "/centerHeightAboveGround";
    float centerHeightAboveGround = icl_core::config::getDefault<float>(stringStream.str(), 0.0f);
    stringStream.str("");
    stringStream << "/classification/class_" << i << "/distanceOfCenters";
    float distanceOfCenters = icl_core::config::getDefault<float>(stringStream.str(), 0.0f);
    stringStream.str("");
    stringStream << "/classification/class_" << i << "/directionOfCentersParallelToGroundThreshold";
    float directionOfCentersParallelToGroundThreshold = icl_core::config::getDefault<float>(stringStream.str(), 0.0f);
    stringStream.str("");
    stringStream << "/classification/class_" << i << "/directionOfCentersUpThreshold";
    float directionOfCentersUpThreshold = icl_core::config::getDefault<float>(stringStream.str(), 0.0f);
    stringStream.str("");
    stringStream << "/classification/class_" << i << "/areaMinThreshold";
    float areaMinThreshold = icl_core::config::getDefault<float>(stringStream.str(), 0.0f);
    stringStream.str("");
    stringStream << "/classification/class_" << i << "/areaMaxThreshold";
    float areaMaxThreshold = icl_core::config::getDefault<float>(stringStream.str(), 1.0f);
    stringStream.str("");
    stringStream << "/classification/class_" << i << "/volumeMinThreshold";
    float volumeMinThreshold = icl_core::config::getDefault<float>(stringStream.str(), 0.0f);
    stringStream.str("");
    stringStream << "/classification/class_" << i << "/volumeMaxThreshold";
    float volumeMaxThreshold = icl_core::config::getDefault<float>(stringStream.str(), 1.0f);
    stringStream.str("");
    stringStream << "/classification/class_" << i << "/shouldBeTall";
    bool shouldBeTall = icl_core::config::getDefault<bool>(stringStream.str(), false);
    stringStream.str("");
    stringStream << "/classification/class_" << i << "/verticalRatioMinThreshold";
    float verticalRatioMinThreshold = icl_core::config::getDefault<float>(stringStream.str(), 0.0f);
    stringStream.str("");
    stringStream << "/classification/class_" << i << "/shouldBeQuadratic";
    bool shouldBeQuadratic = icl_core::config::getDefault<bool>(stringStream.str(), false);
    stringStream.str("");
    stringStream << "/classification/class_" << i << "/horizontalQuadraticThreshold";
    float horizontalQuadraticThreshold = icl_core::config::getDefault<float>(stringStream.str(), 0.0f);
    stringStream.str("");
    stringStream << "/classification/class_" << i << "/densityMinThreshold";
    float densityMinThreshold = icl_core::config::getDefault<float>(stringStream.str(), 0.0f);
    stringStream.str("");
    stringStream << "/classification/class_" << i << "/densityMaxThreshold";
    float densityMaxThreshold = icl_core::config::getDefault<float>(stringStream.str(), 1.0f);
    stringStream.str("");

    gpuclass::ClassificationClass tempClass(name, meaning, parallelToGroundMax, parallelToGroundMin, centerHeightAboveGround, distanceOfCenters,
                                            directionOfCentersParallelToGroundThreshold, directionOfCentersUpThreshold,
                                            areaMinThreshold, areaMaxThreshold,
                                            volumeMinThreshold, volumeMaxThreshold,
                                            shouldBeTall, verticalRatioMinThreshold, shouldBeQuadratic, horizontalQuadraticThreshold,
                                            densityMinThreshold, densityMaxThreshold);
    classes.push_back(tempClass);
  }

  filter->setClassificationClasses(classes);

  nvelo::VelodyneToPointCloudFilter::Ptr velodynePointcloudFilter(new nvelo::VelodyneToPointCloudFilter("velodyne+scaninterpreter:" + velodyneConfFile,"veloPointcloudFilter"));
  //icss::DataSourceBase::Ptr filter;


  //    create the sink
  //    nibeo::PointcloudSinkGpuvoxel::Ptr gpu_voxel_sink;
  //    icss::DataSink<std::vector<gpu_voxels::Vector3f> >::Ptr sink;

  try
  {
    //sink = manager.createSink<std::vector<gpu_voxels::Vector3f> >("pointcloud+gpuvoxel:" , "pointcloudsink");
  }
  catch(const std::exception& e)
  {
    std::cerr << "the sink could not be created!" << std::endl
              << "  " << e.what() << std::endl;
    return 1;
  }

  manager.setSynchronizationPolicy(icss::SP_NO_DATA_DROP);
  manager.seekToBegin();
  manager.printStatus();
  manager.seekRelative(7200);

  for(; manager.good(); manager.advance())
  {
    LOGGING_DEBUG(gpu_voxels::classification::Classification, "started new iteration: " << manager.getCurrentTimestamp() << gpu_voxels::classification::endl);
    nibeo::IbeoMsgStamped::Ptr ibeo_element;
    nncom::NcomStamped::Ptr ncom_element;
    nvelo::PacketFrameStamped::Ptr velodyne_element;

    //read from all sources
    ibeo_element = ibeo_source->get();
    ncom_element = ncom_source->get();
    velodyne_element = velodyne_source->get();


    //read from velodyne
    icl_core::Stamped<VelodynePCLPointCloud::PointCloud>::Ptr velodyneFiltered(new icl_core::Stamped<VelodynePCLPointCloud::PointCloud>());
    velodynePointcloudFilter->filter(*velodyne_element, *velodyneFiltered);


    //        VelodynePCLPointCloud::PointCloud temp = velodyneFiltered->get();
    //        printf("VelodyneSize: %d Velodyne hasChanged: %d Ibeo hasChanged: %d Ncom hasChanged: %d Managed Sources %d %d\n", (temp.width * temp.height),
    //               manager.hasChanged(velodyne_source),
    //               manager.hasChanged(ibeo_source),
    //               manager.hasChanged(ncom_source),
    //               (int)manager.getNumberOfManagedSources(),
    //               (manager.getFirstTimestamp() <= manager.getLastTimestamp()));


    //        icl_core::Stamped<VelodynePCLPointCloud::PointCloud> filtered_stamped;
    //        velodynePointcloudFilter->filter(*velodyne_element, filtered_stamped);

    //filter with both sources
    icl_core::Stamped<std::vector<gpu_voxels::Vector3f> >::Ptr filtered(new icl_core::Stamped<std::vector<gpu_voxels::Vector3f> >());
    filter->setNcomData(ncom_element);

    filter->setVelodynePointCloudData(velodyneFiltered);
    //        filter->setVelodynePointCloudDataStamped(filtered_stamped);

    filter->filter(*ibeo_element, *filtered);


    //output to sink
    //sink->set(filtered);

    //        manager.process();

    manager.printSmallStatus();

    if(stopPreview)
    {
      break;
    }
  }

  LOGGING_DEBUG(gpu_voxels::classification::Classification, "stopped visual preview" << gpu_voxels::classification::endl);

  while(true)
  {
    filter->processPointcloud();
    filter->redrawPointcloud();
  }
}
