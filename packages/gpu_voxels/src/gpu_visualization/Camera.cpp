// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
//
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE.txt in the top
// directory of the source code.
//
// © Copyright 2014 FZI Forschungszentrum Informatik, Karlsruhe, Germany
//
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Matthias Wagner
 * \date    2013-12-2
 *
 *  \brief Camera class for the voxel map visualizer on GPU
 * This class uses a right hand coordinate system.
 * In the world the Z axis points upwards.
 * The camera looks into the direction of positive X. Z points upwards.
 *
 */
//----------------------------------------------------------------------
#include "Camera.h"


namespace gpu_voxels {
namespace visualization {

Camera_gpu::Camera_gpu(float window_width, float window_height, CameraContext context)
{
  m_window_width = window_width;
  m_window_height = window_height;
  m_init_context = context;
  m_speed = .4f;
  m_mouse_speed = 0.00005f;
  m_camera_orbit = false;
  m_mouse_old_x = m_window_width / 2;
  m_mouse_old_y = m_window_height / 2;
  m_has_view_changed = true;
  resetToInitialValues();
}

Camera_gpu::~Camera_gpu()
{
}

void Camera_gpu::resetToInitialValues()
{
  m_cur_context = m_init_context;
  updateCameraDirection();
  updateCameraRight();
  updateProjectionMatrix();
  updateViewMatrix();
}

void Camera_gpu::resizeWindow(float width, float height)
{
  m_window_height = height;
  m_window_width = width;
  updateProjectionMatrix();
}

/**
 * Update the view matrix of the camera.
 *
 * @param xpos: the x coordinate of the mouse pointer.
 * @param ypos: the y coordinate of the mouse pointer.
 */
void Camera_gpu::updateViewMatrixFromMouseInput(int32_t xpos, int32_t ypos)
{

  if (m_camera_orbit)
  {
    ypos = m_window_height - ypos;
    float dx, dy;
    dx = (float) (xpos - m_mouse_old_x);
    dy = (float) (m_mouse_old_y - ypos);

    m_mouse_old_x = xpos;
    m_mouse_old_y = ypos;
    float pitch = dy * 0.001f;
    float yaw = dx * 0.001f;

    yaw = glm::mod(yaw, 2 * float(M_PI)); /*  [0, 2pi]    */
    pitch = std::min(0.4999f * float(M_PI), std::max(-0.4999f * float(M_PI), pitch));/*  (-pi/2,  pi/2)    */

    glm::vec3 right = glm::normalize(m_camera_right);
    glm::vec3 target = getCameraTarget();
    glm::vec3 direction = glm::normalize(target - getCameraPosition());
    glm::vec3 up = glm::normalize(glm::cross(direction, right));
    glm::vec3 camFocusVector = getCameraPosition() - target;

    glm::mat4 rotation1 = glm::rotate(glm::mat4(1), yaw, up);
    glm::mat4 rotation2 = glm::rotate(glm::mat4(1), -pitch, right);

    glm::vec3 camFocusRotated = glm::vec3(rotation1 * rotation2 * glm::vec4(camFocusVector, 0));
    glm::vec3 newCamPos = target + camFocusRotated;

    direction = -glm::normalize(camFocusRotated);
    up = glm::vec3(0, 0, 1);
    right = glm::cross(direction, up);

    m_camera_right = right;
    m_camera_direction = direction;
    m_cur_context.camera_position = newCamPos;

    updateViewMatrix();

  }
  else
  {
    m_cur_context.h_angle += m_mouse_speed * float(m_window_width / 2 - xpos);
    m_cur_context.v_angle += m_mouse_speed * float(m_window_height / 2 - ypos);

    m_cur_context.h_angle = glm::mod(m_cur_context.h_angle, 2 * float(M_PI)); /*  [0, 2pi]    */
    m_cur_context.v_angle = std::min(0.4999f * float(M_PI),
                                     std::max(-0.4999f * float(M_PI), m_cur_context.v_angle));/*  (-pi/2,  pi/2)    */

    updateCameraDirection();
    updateCameraRight();
    updateViewMatrix();
  }
}
void Camera_gpu::setCameraTarget(glm::vec3 camera_target)
{
  m_cur_context.camera_target = camera_target;
  updateViewMatrix();
}

void Camera_gpu::moveAlongDirection(float factor)
{
  m_cur_context.camera_position += m_camera_direction * m_speed * factor;
  updateViewMatrix();
}

void Camera_gpu::moveAlongRight(float factor)
{
  if (!m_camera_orbit)
  {
    m_cur_context.camera_position += m_camera_right * m_speed * factor;
    updateViewMatrix();
  }

}

void Camera_gpu::moveAlongUp(float factor)
{
  if (!m_camera_orbit)
  {
    glm::vec3 up = glm::cross(m_camera_right, m_camera_direction);
    m_cur_context.camera_position += up * m_speed * factor;
    updateViewMatrix();
  }
}

void Camera_gpu::moveFocusPointFromMouseInput(int32_t xpos, int32_t ypos)
{
  ypos = m_window_height - ypos;
  float du, dv;

  glm::vec3 camera_direction = getCameraDirection();
  float horizontal_angle = atan2(camera_direction.y, camera_direction.x);
  float c = cos( horizontal_angle);
  float s = sin( horizontal_angle);
  du = (float) (xpos - m_mouse_old_x) * s + (float) (ypos - m_mouse_old_y) * c;
  dv = (float) (xpos - m_mouse_old_x) * -c + (float) (ypos - m_mouse_old_y) * s;

  m_mouse_old_x = xpos;
  m_mouse_old_y = ypos;

  glm::vec3 target = getCameraTarget();
  target = target - glm::vec3(du, dv, 0);
  setCameraTarget(target);
  glm::vec3 position = getCameraPosition();
  position = position - glm::vec3(du, dv, 0);
  m_cur_context.camera_position = position;
  updateViewMatrix();
}

void Camera_gpu::moveFocusPointVerticalFromMouseInput(int32_t xpos, int32_t ypos)
{
  ypos = m_window_height - ypos;
  float dz;

  dz = (float) (m_mouse_old_y - ypos);

  m_mouse_old_x = xpos;
  m_mouse_old_y = ypos;

  glm::vec3 target = getCameraTarget();
  target = target - glm::vec3(0, 0, dz);
  setCameraTarget(target);
  glm::vec3 position = getCameraPosition();
  position = position - glm::vec3(0, 0, dz);
  m_cur_context.camera_position = position;
  updateViewMatrix();
}


void Camera_gpu::updateViewMatrix()
{
  glm::mat4 view;
  if (m_camera_orbit)
  {
    // Up vector is determined by right vector and view direction
    glm::vec3 up = glm::cross(m_camera_right, m_camera_direction);
    // Cam looks at target point
    view = glm::lookAt(m_cur_context.camera_position, m_cur_context.camera_target, up);
  }
  else
  {
    // Up vector is determined by right vector and view direction
    glm::vec3 up = glm::cross(m_camera_right, m_camera_direction);
    view = glm::lookAt(m_cur_context.camera_position, // Camera is here
        m_cur_context.camera_position + m_camera_direction, // and looks here : at the same position, plus "direction"
        up // Head is up
        );
  }
  m_view_matrix = view;
  m_has_view_changed = true;
}

void Camera_gpu::updateProjectionMatrix()
{
  m_projection_matrix = glm::perspective(m_cur_context.foV, m_window_width / m_window_height, .1f, 5000.0f);
}

void Camera_gpu::updateCameraDirection()
{
  if (m_camera_orbit)
  {
    // direction points from the camera to the target
    m_camera_direction = glm::normalize(m_cur_context.camera_target - m_cur_context.camera_position);
  }
  else
  {
    // in free float mode we pan and tilt the cameras direction vector (X)
    m_camera_direction = glm::normalize(
        glm::vec3(cos(m_cur_context.v_angle) * sin(m_cur_context.h_angle),
                  -cos(m_cur_context.v_angle) * cos(m_cur_context.h_angle),
                  sin(m_cur_context.v_angle)));
  }
}

void Camera_gpu::updateCameraRight()
{
  if (m_camera_orbit)
  {
    // the right vector is determined by the direction and the up vector (Z)
    m_camera_right = glm::cross(m_camera_direction, glm::vec3(0.f, 0.f, 1.f));
  }
  else
  {
    // in free float mode the right vector is only panned around the Z axis
    m_camera_right = glm::vec3(sin(m_cur_context.h_angle - M_PI_2), -cos(m_cur_context.h_angle - M_PI_2), 0 );
  }
}

void Camera_gpu::toggleCameraMode()
{
  m_camera_orbit = !m_camera_orbit;
  updateCameraDirection();
  updateCameraRight();
  updateViewMatrix();
}

std::string Camera_gpu::getCameraInfo()
{
  std::stringstream returnString;
  returnString << "Free Flight Mode: (" << m_cur_context.camera_position.x
               << ", " << m_cur_context.camera_position.y
               << ", " << m_cur_context.camera_position.z << ")\n";
  returnString << "Orbital Mode Focus Point: (" << m_cur_context.camera_target.x
               << ", " << m_cur_context.camera_target.y
               << ", " << m_cur_context.camera_target.z << ")\n";
  returnString << "Horizontal Angle: " << glm::degrees(m_cur_context.h_angle) << "°\n";
  returnString << "Vertical Angle: " << glm::degrees(m_cur_context.v_angle) << "°\n";
  returnString << "Field of View: " << glm::degrees(m_cur_context.foV) << "°\n";
  returnString << "Window Dimensions: W: " << getWindowWidth() << "  H: " << getWindowHeight() << "\n";
  
  return returnString.str();
}


// --- debug functions ---
void Camera_gpu::printCameraPosDirR()
{
  std::cout << "<camera>" << std::endl;
  std::cout << "  <!-- Free flight mode -->" << std::endl;
  std::cout << "  <position>" << std::endl;
  std::cout << "    <x> " << m_cur_context.camera_position.x << " </x>" << std::endl;
  std::cout << "    <y> " << m_cur_context.camera_position.y << " </y>" << std::endl;
  std::cout << "    <z> " << m_cur_context.camera_position.z << " </z>" << std::endl;
  std::cout << "  </position>" << std::endl;
  std::cout << "  <horizontal_angle> " << glm::degrees(m_cur_context.h_angle) << " </horizontal_angle> <!-- given in Deg -->" << std::endl;
  std::cout << "  <vertical_angle> " << glm::degrees(m_cur_context.v_angle) << " </vertical_angle> <!-- given in Deg -->" << std::endl;
  std::cout << "  <field_of_view> " << glm::degrees(m_cur_context.foV) << " </field_of_view> <!-- given in Deg -->" << std::endl;
  std::cout << "  <!-- Orbit mode -->" << std::endl;
  std::cout << "  <focus>" << std::endl;
  std::cout << "    <x> " << m_cur_context.camera_target.x << " </x>" << std::endl;
  std::cout << "    <y> " << m_cur_context.camera_target.y << " </y>" << std::endl;
  std::cout << "    <z> " << m_cur_context.camera_target.z << " </z>" << std::endl;
  std::cout << "  </focus>" << std::endl;
  std::cout << "  <window_width> " << getWindowWidth() << " </window_width>" << std::endl;
  std::cout << "  <window_height> " << getWindowHeight() << " </window_height>" << std::endl;
  std::cout << "</camera>" << std::endl;
}


} // end of namespace visualization
} // end of namespace gpu_voxels
