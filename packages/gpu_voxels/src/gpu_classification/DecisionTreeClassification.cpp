#include "DecisionTreeClassification.h"

#include <thrust/device_ptr.h>

namespace gpu_voxels{
namespace classification{

//Constructor
DecisionTreeClassification::DecisionTreeClassification()
{

}

void DecisionTreeClassification::setClasses(std::vector<ClassificationClass> classes)
{
  //display all classes in LOG
  for(uint i = 0; i < classes.size(); i++)
  {
    LOGGING_DEBUG(DecisionTree, " Class: " << classes[i].name);
  }
  LOGGING_DEBUG(DecisionTree, endl);
  //register classes
  this->classes = classes;
}

void DecisionTreeClassification::classifySegments(std::vector<Segment> segments)
{
  //store segments
  initialize(segments);

  for(uint i = 0; i < classes.size(); i++)
  {
    LOGGING_INFO(DecisionTree, " Class: " << classes[i].name);
  }
  LOGGING_INFO(DecisionTree, endl);

  //check each segment
  for(uint i = 0; i < this->segments.size(); i++)
  {
    //check segment i against all classes
    checkForClass(i);
  }
}

std::vector<Segment> DecisionTreeClassification::getClassifiedSegments()
{
  return segments;
}

int DecisionTreeClassification::getClassifiedSegmentsSize()
{
  return segments.size();
}

void DecisionTreeClassification::initialize(std::vector<Segment> segments)
{
  this->segments = segments;
}


void DecisionTreeClassification::checkForClass(int segmentIndex)
{
  //used to determine whether fallback is needed or not
  bool foundMatch = false;

  //check all classes
  for(uint i = 0; i < this->classes.size(); i++)
  {
    //if segment is member of class
    if(this->classes[i].isSegmentInClass(this->segments[segmentIndex]))
    {
      //set the bitvoxelmeaning of the segment
      this->segments[segmentIndex].setMeaning(this->classes[i].meaning);
      foundMatch = true;
    }
  }

  //fallback meaning
  if(!foundMatch)
  {
    this->segments[segmentIndex].setMeaning(static_cast<gpu_voxels::BitVoxelMeaning>(19));
  }
}


}//end namespace classification
}//end namespace gpu_voxels
