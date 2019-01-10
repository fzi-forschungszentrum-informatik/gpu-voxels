// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Sebastian Klemm
 * \date    2012-09-13
 *
 */
//----------------------------------------------------------------------
#include "KinematicChain.h"
#include "gpu_voxels/robot/kernels/KinematicOperations.h"

#include <iostream>
#include <string>
#include <map>
#include <utility>

namespace gpu_voxels {
namespace robot {

KinematicChain::KinematicChain(const std::vector<std::string> &linknames,
                const std::vector<robot::DHParameters> &dh_params,
                const MetaPointCloud &pointclouds)
{
  m_links_meta_cloud = new MetaPointCloud(pointclouds);
  init(linknames, dh_params);
}



KinematicChain::KinematicChain(const std::vector<std::string> &linknames,
                               const std::vector<robot::DHParameters> &dh_params,
                               const std::vector<std::string> &paths_to_pointclouds,
                               const bool use_model_path)
{
  // The untransformed point clouds
  m_links_meta_cloud = new MetaPointCloud(paths_to_pointclouds, paths_to_pointclouds, use_model_path);
  init(linknames, dh_params);
}

void KinematicChain::init(const std::vector<std::string> &linknames,
                          const std::vector<robot::DHParameters> &dh_params)
{
  m_linknames = linknames;
  HANDLE_CUDA_ERROR(cudaEventCreate(&m_start));
  HANDLE_CUDA_ERROR(cudaEventCreate(&m_stop));

  // sanity check:
  if (linknames.size() != dh_params.size())
  {
    LOGGING_ERROR_C(RobotLog, KinematicChain,
                    "Number of linknamesames does not fit number of DH parameters. EXITING!" << endl);
    exit(-1);
  }
//  std::map<uint16_t, std::string> cloud_names = m_links_meta_cloud->getCloudNames();
//  for (size_t i = 0; i < linknames.size(); i++)
//  {
//    if(cloud_names[i] != linknames[i])
//    {
//      LOGGING_ERROR_C(RobotLog, KinematicChain,
//                      "Names of clouds differ from names of links. EXITING!" << endl);
//      exit(-1);
//    }
//  }

  for(size_t i = 0; i < dh_params.size(); i++)
  {
    m_links[linknames[i]] = KinematicLinkSharedPtr(new KinematicLink(dh_params[i]));
  }


  LOGGING_INFO_C(RobotLog, KinematicChain, "now handling " << m_links.size() << " links." << endl);

  // allocate a copy of the pointclouds to store the transformed clouds (host and device)
  m_transformed_links_meta_cloud = new MetaPointCloud(*m_links_meta_cloud);

}

KinematicChain::~KinematicChain()
{
  // destroy the copy of the transformed meta cloud on host and device:
  delete m_links_meta_cloud;
  delete m_transformed_links_meta_cloud;

  HANDLE_CUDA_ERROR(cudaEventDestroy(m_start));
  HANDLE_CUDA_ERROR(cudaEventDestroy(m_stop));
}

void KinematicChain::setConfiguration(const JointValueMap &jointmap)
{
  for (std::map<std::string, KinematicLinkSharedPtr>::const_iterator it=m_links.begin();
       it!=m_links.end(); ++it)
  {
    if(jointmap.count(it->first) != 0)
    {
      it->second->setJointValue(jointmap.at(it->first));
    }
  }

  Matrix4f transformation = gpu_voxels::Matrix4f::createIdentity();

  // Iterate over all links and transform pointclouds with the according name
  // if no pointcloud was found, still the transformation has to be calculated and copied to the device
  // for the next link.
  for(size_t i = 0; i < m_linknames.size(); i++)
  {
    std::string linkname = m_linknames[i];
    if (m_links_meta_cloud->hasCloud(linkname))
    {
      int16_t pc_num = m_links_meta_cloud->getCloudNumber(linkname);
      m_links_meta_cloud->transformSubCloud(pc_num, &transformation, m_transformed_links_meta_cloud);
    }
    // Sending the actual transformation for this link to the GPU.
    // This means the DH Transformation i is not applied to link-pointcloud i,
    // but to link poincloud i+1, i+2...
    m_links[linkname]->getMatrixRepresentation(m_dh_transformation);
    transformation = transformation * m_dh_transformation;
    //std::cout << "Trafo Matrix ["<< linkname <<"] = " << m_dh_transformation  << std::endl;
    //std::cout << "Accumulated Trafo Matrix ["<< linkname <<"] = " << transformation << std::endl;
  }
  cudaDeviceSynchronize();
}


void KinematicChain::getJointNames(std::vector<std::string> &jointnames)
{
  jointnames.clear();
  for (std::map<std::string, KinematicLinkSharedPtr>::const_iterator it=m_links.begin();
       it!=m_links.end(); ++it)
  {
    jointnames.push_back(it->first);
  }
}

void KinematicChain::getConfiguration(JointValueMap &jointmap)
{
  for (std::map<std::string, KinematicLinkSharedPtr>::const_iterator it=m_links.begin();
       it!=m_links.end(); ++it)
  {
    jointmap[it->first] = it->second->getJointValue();
  }
}

void KinematicChain::getLowerJointLimits(JointValueMap &lower_limits)
{
  LOGGING_ERROR_C(RobotLog, KinematicChain,
                  "getLowerJointLimits not implemented for DH-Robot!" << endl);
}

void KinematicChain::getUpperJointLimits(JointValueMap &upper_limits)
{
  LOGGING_ERROR_C(RobotLog, KinematicChain,
                  "getUpperJointLimits not implemented for DH-Robot!" << endl);
}

/* private helper functions */

//void KinematicChain::transformPointAlongChain(Vector3f point)
//{
//  Vector3f transformed_point;

//  Vector3f* dev_point = NULL;
//  Vector3f* dev_result = NULL;
//  HANDLE_CUDA_ERROR(cudaMalloc(&dev_point, sizeof(Vector3f)));
//  HANDLE_CUDA_ERROR(cudaMalloc(&dev_result, sizeof(Vector3f)));

//  for (unsigned int i = 0; i <= m_size; i++)
//  {
//    HANDLE_CUDA_ERROR(cudaMemcpy(dev_point, &point, sizeof(Vector3f), cudaMemcpyHostToDevice));
//    kernelTransformPoseAlongChain<<< 1, 1 >>>
//    (m_size, (i), m_dev_basis_transformation,
//        thrust::raw_pointer_cast(&(m_dev_transformations[0])),
//        dev_point, dev_result);
//    CHECK_CUDA_ERROR();

//    HANDLE_CUDA_ERROR(cudaMemcpy(&transformed_point, dev_result, sizeof(Vector3f), cudaMemcpyDeviceToHost));
//    std::stringstream s;
//    s << "transformation around joint i=" << i << " : " << (transformed_point);
//    LOGGING_DEBUG_C(RobotLog, KinematicChain,  s.str()  << endl);
//  }

//  HANDLE_CUDA_ERROR(cudaFree(dev_result));
//  HANDLE_CUDA_ERROR(cudaFree(dev_point));

//}

void KinematicChain::updatePointcloud(const std::string &link_name, const std::vector<Vector3f> &cloud)
{
  m_links_meta_cloud->updatePointCloud(link_name, cloud, true);
}

} // end of namespace
} // end of namespace
