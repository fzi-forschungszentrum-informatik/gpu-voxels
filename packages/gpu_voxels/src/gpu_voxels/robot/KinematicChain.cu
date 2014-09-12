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
#include "kernels/KinematicOperations.h"

#include <iostream>
#include <string>
#include <map>
#include <utility>

namespace gpu_voxels {

KinematicChain::KinematicChain(const std::vector<KinematicLinkSharedPtr> &links, const MetaPointCloud &robot_meta_cloud,
                               Matrix4f basis_transformation, std::map<unsigned int, std::string>* linknames) :
    m_size(links.size())
{
  // generate a local copy of the untransformed point cloud
  m_links_meta_cloud = new MetaPointCloud(robot_meta_cloud);

  HANDLE_CUDA_ERROR(cudaEventCreate(&m_start));
  HANDLE_CUDA_ERROR(cudaEventCreate(&m_stop));

  m_basis_transformation = basis_transformation;
  m_links = links;

  // sanity check:
  if (links.size() != m_links_meta_cloud->getNumberOfPointclouds())
  {
    LOGGING_ERROR_C(RobotLog, KinematicChain,
                    "Number of PointClouds does not fit number if links. EXITING!" << endl);
    exit(-1);
    if (linknames && (links.size() != linknames->size()))
    {
      LOGGING_ERROR_C(RobotLog, KinematicChain,
                      "Number of LinkNames does not fit number of links. EXITING!" << endl);

      exit(-1);
    }
  }

  LOGGING_INFO_C(RobotLog, KinematicChain, "now handling " << m_size << " links." << endl);
  m_dh_parameters.resize(m_size);

  m_dev_transformations.resize(m_size);
  m_dev_local_transformations.resize(m_size);

  m_joint_types.resize(m_size);
  m_dev_joint_types.resize(m_size);

  for (uint8_t i = 0; i < m_size; i++)
  {
    m_dh_parameters[i] = m_links[i]->getDHParam();
    m_joint_types[i] = (uint8_t) m_links[i]->getJointType();
  }
  m_dev_joint_types = m_joint_types;

  // allocate a copy of the pointclouds to store the transformed clouds (host and device)
  m_transoformed_links_meta_cloud = new MetaPointCloud(m_links_meta_cloud);

  HANDLE_CUDA_ERROR(cudaMalloc((void** )&m_dev_basis_transformation, sizeof(Matrix4f)));
  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_dev_basis_transformation, &m_basis_transformation, sizeof(Matrix4f),
                 cudaMemcpyHostToDevice));
}

KinematicChain::~KinematicChain()
{
  HANDLE_CUDA_ERROR(cudaFree(m_dev_basis_transformation));

  // destroy the copy of the transformed meta cloud on host and device:
  delete m_transoformed_links_meta_cloud;

  HANDLE_CUDA_ERROR(cudaEventDestroy(m_start));
  HANDLE_CUDA_ERROR(cudaEventDestroy(m_stop));
}

void KinematicChain::update()
{
//  updateJointValues();
  Matrix4f transformation;
  transformation.setIdentity();
  transformation = transformation * m_basis_transformation;

  Matrix4f* transformation_dev;
  HANDLE_CUDA_ERROR(cudaMalloc((void** )&transformation_dev, sizeof(Matrix4f)));
  HANDLE_CUDA_ERROR(
      cudaMemcpy(transformation_dev, &transformation, sizeof(Matrix4f), cudaMemcpyHostToDevice));

  for (uint8_t i = 0; i < m_size; ++i)
  {
    m_math.computeLinearLoad(m_links_meta_cloud->getPointcloudSize(i), &m_blocks, &m_threads_per_block);
//    printf("for joint %u: blocks = %u, threads = %u\n", i, m_blocks, m_threads_per_block);
//    printf("    to address %u points in cloud.\n", m_point_cloud_sizes[i]);
    //First inserting the Link with the current transformation
    cudaDeviceSynchronize();
    kernelKinematicChainTransform<<< m_blocks, m_threads_per_block >>>
    (m_size, i, transformation_dev,
        m_links_meta_cloud->getDeviceConstPointer(),
        m_transoformed_links_meta_cloud->getDevicePointer());
    Matrix4f dh_transformation;

    // Sending the actual transformation for this link to the GPU.
    // This means the Transformation for link i is not applied to link i, but to link i+1, i+2...

    KinematicLink::DHParameters dh_parameter = m_links[i]->getDHParam();
    convertDHtoMHost(dh_parameter.theta, dh_parameter.d,
                     0, // currently only b = 0
                     dh_parameter.a, dh_parameter.alpha, dh_parameter.value,
                     (uint8_t) m_links[i]->getJointType(), dh_transformation);
    transformation = transformation * dh_transformation;
    HANDLE_CUDA_ERROR(
        cudaMemcpy(transformation_dev, &transformation, sizeof(Matrix4f), cudaMemcpyHostToDevice));
  }
  cudaDeviceSynchronize();
  HANDLE_CUDA_ERROR(cudaFree(transformation_dev));
}

void KinematicChain::setConfiguration(Matrix4f basis_transformation, std::vector<float> joint_values)
{
  m_basis_transformation = basis_transformation;
  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_dev_basis_transformation, &m_basis_transformation, sizeof(Matrix4f),
                 cudaMemcpyHostToDevice));
  setConfiguration(joint_values);
}

void KinematicChain::setConfiguration(std::vector<float> joint_values)
{
//  std::cout << "setConfig: joint_values.size = " << joint_values.size() << std::endl;
//  std::cout << "setConfig: m_size = " << m_size << std::endl;

  if (m_size != joint_values.size())
  {
    LOGGING_ERROR_C(RobotLog, KinematicChain, "Kinematic chain size != number of joints" << endl);
    return;
  }

  for (uint8_t i = 0; i < m_size; i++)
  {
    m_links[i]->setJointValue(joint_values[i]);
  }
  update();
}

void KinematicChain::convertDHtoMHost(float theta, float d, float b, float a, float alpha, float q,
                                      uint8_t joint_type, Matrix4f& m)
{
//  printf("theta, d, a, alpha : \t%f, %f, %f, %f\n", theta, d, a, alpha);
  float ca = 0;
  float sa = 0;
  float ct = 0;
  float st = 0;

  if (joint_type == KinematicLink::PRISMATIC) /* Prismatic joint */
  {
    d += q;
  }
  else /* Revolute joint */
  {
    if (joint_type != KinematicLink::REVOLUTE)
    {
      LOGGING_ERROR_C(RobotLog, KinematicChain, "Illegal joint type" << endl);
    }
    theta += q;
  }

  ca = (float) cos(alpha);
  sa = (float) sin(alpha);
  ct = (float) cos(theta);
  st = (float) sin(theta);

  m.a11 = ct;
  m.a12 = -st * ca;
  m.a13 = st * sa;
  m.a14 = a * ct - b * st;

  m.a21 = st;
  m.a22 = ct * ca;
  m.a23 = -ct * sa;
  m.a24 = a * st + b * ct;

  m.a31 = 0.0;
  m.a32 = sa;
  m.a33 = ca;
  m.a34 = d;

  m.a41 = 0.0;
  m.a42 = 0.0;
  m.a43 = 0.0;
  m.a44 = 1.0;
}

/* private helper functions */
void KinematicChain::updateJointValues()
{
  for (uint8_t i = 0; i < m_size; i++)
  {
    m_dh_parameters[i] = m_links[i]->getDHParam();
  }
}

void KinematicChain::transformPointAlongChain(Vector3f point)
{
  Vector3f transformed_point;

  Vector3f* dev_point = NULL;
  Vector3f* dev_result = NULL;
  HANDLE_CUDA_ERROR(cudaMalloc(&dev_point, sizeof(Vector3f)));
  HANDLE_CUDA_ERROR(cudaMalloc(&dev_result, sizeof(Vector3f)));

  for (unsigned int i = 0; i <= m_size; i++)
  {
    HANDLE_CUDA_ERROR(cudaMemcpy(dev_point, &point, sizeof(Vector3f), cudaMemcpyHostToDevice));
    kernelTransformPoseAlongChain<<< 1, 1 >>>
    (m_size, (i), m_dev_basis_transformation,
        thrust::raw_pointer_cast(&(m_dev_transformations[0])),
        dev_point, dev_result);

    HANDLE_CUDA_ERROR(cudaMemcpy(&transformed_point, dev_result, sizeof(Vector3f), cudaMemcpyDeviceToHost));
    std::stringstream s;
    s << "transformation around joint i=" << i << " : " << (transformed_point);
    LOGGING_DEBUG_C(RobotLog, KinematicChain,  s.str()  << endl);
  }

  HANDLE_CUDA_ERROR(cudaFree(dev_result));
  HANDLE_CUDA_ERROR(cudaFree(dev_point));

}

void KinematicChain::updatePointcloud(uint16_t link, const std::vector<Vector3f> &cloud)
{
  m_links_meta_cloud->updatePointCloud(link, cloud, true);
}

} // end of namespace
