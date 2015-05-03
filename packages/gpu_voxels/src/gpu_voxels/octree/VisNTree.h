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
 * \date    2014-06-18
 *
 */
//----------------------------------------------------------------------/*
#ifndef VISNTREE_H_
#define VISNTREE_H_

#include <gpu_voxels/octree/Octree.h>
#include <gpu_voxels/vis_interface/VisProvider.h>
#include <gpu_voxels/helpers/cuda_handling.h>

namespace gpu_voxels {
namespace NTree {

template<typename InnerNode, typename LeafNode>
class VisNTree: public VisProvider
{
public:
  typedef NTree<8, 15, InnerNode, LeafNode> MyNTree;

  VisNTree(MyNTree* ntree, std::string map_name);

  virtual ~VisNTree();

  virtual bool visualize(const bool force_repaint = true);

  virtual uint32_t getResolutionLevel();

protected:
  MyNTree* m_ntree;
  cudaIpcMemHandle_t* m_shm_memHandle;
  uint32_t m_min_level; // Min visualization level of octree of last visualize() call
  uint32_t* m_shm_superVoxelSize;
  uint32_t* m_shm_numCubes;
  bool* m_shm_bufferSwapped;
  bool m_internal_buffer_1;
  thrust::device_vector<Cube> *m_d_cubes_1;
  thrust::device_vector<Cube> *m_d_cubes_2;
};
}

}

#endif /* VISNTREE_H_ */
