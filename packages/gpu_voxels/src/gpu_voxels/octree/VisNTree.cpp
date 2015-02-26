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
 * \author  Andreas Hermann
 * \date    2015-02-26
 *
 * This is just a helper class that is used to force the compiler
 * to generate an object file for the templated VisNTree
 *
 */
//----------------------------------------------------------------------

#include <gpu_voxels/octree/VisNTree.hpp>

namespace gpu_voxels {
namespace NTree {

template class VisNTree<Environment::InnerNodeProb, Environment::LeafNodeProb>;
template class VisNTree<Environment::InnerNode, Environment::LeafNode>;

}
}
