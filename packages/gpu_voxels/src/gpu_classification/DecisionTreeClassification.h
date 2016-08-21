#ifndef DECISIONTREECLASSIFICATION_H
#define DECISIONTREECLASSIFICATION_H

#include <gpu_classification/ObjectSegmentationHelper.h>

namespace gpu_voxels{
namespace classification{

//Represents a class for classification (like car, tree...)
//A class is defined by its parameters. Only if a segment passes all checks
//against the parameters, it is considered a member of this class.
struct ClassificationClass
{
  std::string name;

  //needed for coloring in the gpu_visualizer
  gpu_voxels::BitVoxelMeaning meaning;

  //normal check properties
  float parallelToGroundMax;
  float parallelToGroundMin;

  float centerHeightAboveGround;

  //center check properties
  float distanceOfCenters;
  float directionOfCentersParallelToGroundThreshold;
  float directionOfCentersUpThreshold;

  //area check properties
  float areaMinThreshold;
  float areaMaxThreshold;

  //volume check properties
  float volumeMinThreshold;
  float volumeMaxThreshold;

  //edgeratios check properties
  bool shouldBeTall;
  float verticalRatioMinThreshold;
  bool shouldBeQuadratic;
  float horizontalQuadraticThreshold;

  //density check properties
  float densityMinThreshold;
  float densityMaxThreshold;


  //Constructor
  ClassificationClass(std::string name, int meaning, float pTGTMax, float pTGTMin, float cHAG, float dOC, float dOCPTGT, float dOCUT, float aMinT, float aMaxT, float vMinT, float vMaxT, bool sBT, float vRMinT, bool sBQ, float hQT, float dMinT, float dMaxT)
  {
    this->name = name;
    this->meaning = static_cast<gpu_voxels::BitVoxelMeaning>(meaning);
    this->parallelToGroundMax = pTGTMax;
    this->parallelToGroundMin = pTGTMin;
    this->centerHeightAboveGround = cHAG;
    this->distanceOfCenters = dOC;
    this->directionOfCentersParallelToGroundThreshold = dOCPTGT;
    this->directionOfCentersUpThreshold = dOCUT;
    this->areaMinThreshold = aMinT;
    this->areaMaxThreshold = aMaxT;
    this->volumeMinThreshold = vMinT;
    this->volumeMaxThreshold = vMaxT;
    this->shouldBeTall = sBT;
    this->verticalRatioMinThreshold = vRMinT;
    this->shouldBeQuadratic = sBQ;
    this->horizontalQuadraticThreshold = hQT;
    this->densityMinThreshold = dMinT;
    this->densityMaxThreshold = dMaxT;
  }

  /*!
     * \brief isSegmentInClass
     * \param segment segment to check against this class
     * \return true if the segment is a member of this class
     */
  bool isSegmentInClass(Segment segment)
  {
    this->segment = segment;

    //check the whole decision tree
    if(isNormalParallelToGround() &&
       isCenterHeightAboveGround() &&
       (isCenterAlligned() || (isCenterMissallignmentParallelToGround() && isCenterMissallignmentUp())) &&
       isAreaInRange() &&
       isVolumeInRange() &&
       (!shouldBeTall || isTall()) && (!shouldBeQuadratic || isQuadratic()) &&
       isDensityInRange())
    {
      return true;
    }
    return false;
  }

  //used to set all parameters after the construction of an instance
  void setParameters(std::string name, int meaning, float pTGTMax, float pTGTMin, float cHAG, float dOC, float dOCPTGT, float dOCUT, float aMinT, float aMaxT, float vMinT, float vMaxT, bool sBT, float vRMinT, bool sBQ, float hQT, float dMinT, float dMaxT)
  {
    this->name = name;
    this->meaning = static_cast<gpu_voxels::BitVoxelMeaning>(meaning);
    this->parallelToGroundMax = pTGTMax;
    this->parallelToGroundMin= pTGTMin;
    this->centerHeightAboveGround = cHAG;
    this->distanceOfCenters = dOC;
    this->directionOfCentersParallelToGroundThreshold = dOCPTGT;
    this->directionOfCentersUpThreshold = dOCUT;
    this->areaMinThreshold = aMinT;
    this->areaMaxThreshold = aMaxT;
    this->volumeMinThreshold = vMinT;
    this->volumeMaxThreshold = vMaxT;
    this->shouldBeTall = sBT;
    this->verticalRatioMinThreshold = vRMinT;
    this->shouldBeQuadratic = sBQ;
    this->horizontalQuadraticThreshold = hQT;
    this->densityMinThreshold = dMinT;
    this->densityMaxThreshold = dMaxT;
  }

private:

  //used to store the segment to check an use it in the private methods
  Segment segment;

  //check Normal Properties
  bool isNormalParallelToGround()
  {
    Vector3f up(0.0f, 0.0f, 1.0f);
    float scalarProduct = fabsf((up.x * segment.normal.x + up.y * segment.normal.y + up.z * segment.normal.z) - 1);
    if(scalarProduct > parallelToGroundMin && scalarProduct < parallelToGroundMax)
    {
      return true;
    }
    else
    {
      return false;
    }
  }

  bool isCenterHeightAboveGround()
  {
    if(segment.geoCenter.z < centerHeightAboveGround)
    {
      return true;
    }
    return false;
  }

  //check Center Properties
  bool isCenterAlligned()
  {
    Vector3f dCenters = segment.baryCenter - segment.geoCenter;
    float distance = sqrt(dCenters.x * dCenters.x + dCenters.y * dCenters.y + dCenters.z * dCenters.z);
    if(distance < distanceOfCenters)
    {
      return true;
    }
    return false;
  }
  bool isCenterMissallignmentParallelToGround()
  {
    Vector3f dCenters = segment.baryCenter - segment.geoCenter;
    Vector3f up(0.0f, 0.0f, 1.0f);
    float scalarProduct = fabsf((up.x * dCenters.x + up.y * dCenters.y + up.z * dCenters.z) - 1);
    if(scalarProduct < directionOfCentersParallelToGroundThreshold)
    {
      return true;
    }
    return false;

  }
  bool isCenterMissallignmentUp()
  {
    Vector3f dCenters = segment.baryCenter - segment.geoCenter;
    Vector3f up(0.0f, 0.0f, 1.0f);
    float scalarProduct = up.x * dCenters.x + up.y * dCenters.y + up.z * dCenters.z;
    if(scalarProduct < directionOfCentersUpThreshold)
    {
      return true;
    }
    return false;
  }

  //check area properties
  bool isAreaInRange()
  {
    if(segment.xyArea > areaMinThreshold && segment.xyArea < areaMaxThreshold)
    {
      return true;
    }
    return false;
  }

  //check volume properties
  bool isVolumeInRange()
  {
    if(segment.volume > volumeMinThreshold && segment.volume < volumeMaxThreshold)
    {
      return true;
    }
    return false;
  }

  //check edgeratios properties
  bool isTall()
  {
    if(segment.edgeRatios.y > verticalRatioMinThreshold && segment.edgeRatios.z > verticalRatioMinThreshold)
    {
      return true;
    }
    return false;
  }
  bool isQuadratic()
  {
    if(segment.edgeRatios.x - 1 < horizontalQuadraticThreshold)
    {
      return true;
    }
    return false;
  }

  //check density properties
  bool isDensityInRange()
  {
    if(segment.density > densityMinThreshold && segment.density < densityMaxThreshold)
    {
      return true;
    }
    return false;
  }

};

/*!
 * \brief The DecisionTreeClassification class
 */
class DecisionTreeClassification
{
public:
  //Constructor
  DecisionTreeClassification();

  /*!
    * Checks the input segments against all classes given by setClasses.
    * \param segments: a vector of all segments, which should be checked against all classes
    */
  void classifySegments(std::vector<Segment> segments);

  /*!
    * This method is used to initialize the classes which are used.
    * \param classes: a vector of ClassificationClasses which is used to check segments
    */
  void setClasses(std::vector<ClassificationClass> classes);

  /*!
    * \return a vector of segments after classification
    */
  std::vector<Segment> getClassifiedSegments();

  /*!
    * \return size of the vector of segments after classification
    */
  int getClassifiedSegmentsSize();

private:

  //store input segments
  std::vector<Segment> segments;

  //store classes to check against
  std::vector<ClassificationClass> classes;

  /*!
    * \param segments: segments to check with checkForClass
    */
  void initialize(std::vector<Segment> segments);

  /*!
    * \param segmentIndex: index of segment to check against all classes
    */
  void checkForClass(int segmentIndex);
};

}//end namespace classification
}//end namespace gpu_voxels

#endif // DECISIONTREECLASSIFICATION_H
