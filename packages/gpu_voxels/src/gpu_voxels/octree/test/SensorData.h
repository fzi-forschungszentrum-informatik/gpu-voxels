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
 * \author  Florian Drews
 * \date    2014-04-25
 *
 */
//----------------------------------------------------------------------


#ifndef SENSORDATA_H_
#define SENSORDATA_H_

#include <gpu_voxels/octree/test/ArgumentHandling.h>
#include <gpu_voxels/octree/test/Provider.h>

namespace gpu_voxels {
namespace NTree {

class SensorData
{
public:

  SensorData(Provider::Provider* provider, const Provider::Provider_Parameter* parameter) :
      m_provider(provider), m_parameter(parameter)
  {
  }
  virtual ~SensorData()
  {
  }

  virtual void run() = 0;
  virtual void stop() = 0;
  virtual bool isRunning() = 0;
  virtual void takeImage() = 0;

protected:
  Provider::Provider* m_provider;
  const Provider::Provider_Parameter* m_parameter;
};

}
}

#endif /* SENSORDATA_H_ */
