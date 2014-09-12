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
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_VISUALIZATION_CAMERA_GPU_H_INCLUDED
#define GPU_VOXELS_VISUALIZATION_CAMERA_GPU_H_INCLUDED

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>

#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

namespace gpu_voxels {
namespace visualization {

class Camera_gpu
{
public:
  struct CameraContext
  {
    CameraContext()
      : camera_position(0.f, 10.f, 0.f),
        h_angle(0.5f),
        v_angle(0.f),
        foV(60.0f)
    {
      camera_target = glm::vec3(250.f, 0, 250.f);
    }

    CameraContext(glm::vec3 cam_pos, glm::vec3 cam_focus,
                  float horizontal_angle, float vertical_angle, float field_of_view)
    {
      camera_position = cam_pos;
      h_angle = horizontal_angle;
      v_angle = vertical_angle;
      foV = field_of_view;
      camera_target = cam_focus;
    }

    ~CameraContext()
    {
    }

    // the position of the camera
    glm::vec3 camera_position;
    // the point where the camera is looking in orbit mode
    glm::vec3 camera_target;

    // the horizontal, vertical angle and field of view
    float h_angle, v_angle, foV;
  };

  Camera_gpu(float window_width, float window_height, CameraContext context);

  ~Camera_gpu();

  void resetToInitialValues();
  void resizeWindow(float width, float height);
  ///////////////////////////////////// functions for camera movement /////////////////////////////////
  void moveAlongDirection(float factor);
  void moveAlongRight(float factor);
  void moveAlongUp(float factor);

  void moveFocusPointFromMouseInput(int32_t xpos, int32_t ypos);

  void updateViewMatrixFromMouseInput(int32_t xpos, int32_t ypos);

  void toggleCameraMode();

  bool hasViewChanged()
  {
    return m_has_view_changed;
  }

  void setViewChanged(bool view_changed)
  {
    m_has_view_changed = view_changed;
  }

  //////////////////////////////////////// getter / setter /////////////////////////////////////////////
  glm::vec3 getCameraDirection() const
  {
    return m_camera_direction;
  }

  glm::vec3 getCameraPosition() const
  {
    return m_cur_context.camera_position;
  }

  glm::vec3 getCameraTarget() const
  {
    return m_cur_context.camera_target;
  }

  void setCameraTarget(glm::vec3 camera_target);

  void setCameraTargetOfInitContext(glm::vec3 camera_target)
  {
    m_init_context.camera_target = camera_target;
  }

  glm::mat4 getProjectionMatrix() const
  {
    return m_projection_matrix;
  }

  glm::mat4 getViewMatrix() const
  {
    return m_view_matrix;
  }

  float getWindowHeight() const
  {
    return m_window_height;
  }

  void setWindowHeight(float windowHeight)
  {
    m_window_height = windowHeight;
  }

  float getWindowWidth() const
  {
    return m_window_width;
  }

  void setWindowWidth(float windowWidth)
  {
    m_window_width = windowWidth;
  }

  void setMousePosition(int32_t x, int32_t y)
  {
    m_mouse_old_x = x;
    m_mouse_old_y = y;
  }

  // --- debug functions ---
  void printCameraPosDirR();

  void printCameraTargetPointPos();

  glm::vec3 getCameraRight() const
  {
    return m_camera_right;
  }
  // --- end debug functions ---

private:

  void updateViewMatrix();

  void updateProjectionMatrix();

  void updateRotationMatrix();

  void updateCameraDirection();

  void updateCameraRight();

  // the initial context of the camera
  CameraContext m_init_context;
  // the current context of the camera
  CameraContext m_cur_context;

  glm::mat4 m_view_matrix;
  glm::mat4 m_projection_matrix;

  glm::vec3 m_camera_direction;
  glm::vec3 m_camera_right;

  float m_speed;
  float m_mouse_speed;

  float m_window_width;
  float m_window_height;

  bool m_camera_orbit;
  int32_t m_mouse_old_x;
  int32_t m_mouse_old_y;

  bool m_has_view_changed;

};

} // end of namespace visualization
} //end of namespace gpu_voxels
#endif
