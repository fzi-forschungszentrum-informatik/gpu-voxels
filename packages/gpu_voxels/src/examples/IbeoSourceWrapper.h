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
 * \date    2014-10-28
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_HELPERS_IBEO_SOURCE_WRAPPER_H_INCLUDED
#define GPU_VOXELS_HELPERS_IBEO_SOURCE_WRAPPER_H_INCLUDED

#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <string>
#include <vector>
#include <boost/function.hpp>

namespace gpu_voxels {

/**
 * @brief The IbeoSourceWrapper class wraps around the \a IbeoSourceNoAPI and the \a NcomSource to get transformed sensor point clouds.
 */
class IbeoSourceWrapper{

public:
    typedef boost::function<void (std::vector<Vector3f>&)> CallbackFunction;

    /**
     * @brief IbeoSourceWrapper constructor
     * @param callback Callback for new transformed point clouds from the sensor
     * @param ibeo_uri URI used to create the ibeo source.
     * @param ncom_uri URI used to create the ncom source.
     */
    IbeoSourceWrapper(CallbackFunction callback, std::string ibeo_uri, std::string ncom_uri, Vector3f additional_translation = Vector3f());

    /**
     * @brief Start delivering transformed point clouds by the callback function.
     */
    void run();

    /**
     * @brief Stop delivering new point clouds to the callback function.
     */
    void stop();

protected:
    CallbackFunction m_callback;
    std::string m_ibeo_uri, m_ncom_uri;
    bool m_abort;
    Vector3f m_additional_translation;
};

}

#endif
