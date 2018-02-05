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
 * \author  Herbert Pietrzyk
 * \date    2016-05-13
 *
 * This is the general interface definition from which all kinds of maps
 * inherit their collision API.
 *
 */
//----------------------------------------------------------------------/*

#ifndef GPU_VOXELS_HELPERS_COLLISIONINTERFACES_H_INCLUDED
#define GPU_VOXELS_HELPERS_COLLISIONINTERFACES_H_INCLUDED

#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/common_defines.h>

namespace gpu_voxels{


// BITVECTOR VOXELMAP
class CollidableWithBitVectorVoxelMap
{
public:
  /*!
   * \brief collideWith This does a collision check with 'other'.
   * \param map The map to do a collision check with.
   * \param coll_threshold The threshold when a collision is counted. Only valid for probabilistic maps.
   * \return The severity of the collision, namely the number of voxels that lie in collision
   */
  virtual size_t collideWith(const voxelmap::BitVectorVoxelMap* map, float coll_threshold = 1.0, const Vector3i &offset = Vector3i()) = 0;
};

class CollidableWithResolutionBitVectorVoxelMap
{
public:
  /*!
   * \brief collideWithResolution This does a collision check with 'other'.
   * \param map The map to do a collision check with.
   * \param coll_threshold The threshold when a collision is counted. Only valid for probabilistic maps.
   * \param resolution_level The resolution used for collision checking. resolution_level = 0 delivers the highest accuracy whereas each increase haves the resolution.
   * \param offset The offset in cell coordinates
   * \return The severity of the collision, namely the number of voxels that lie in collision
   */
  virtual size_t collideWithResolution(const voxelmap::BitVectorVoxelMap* map, float coll_threshold = 1.0, const uint32_t resolution_level = 0, const Vector3i &offset = Vector3i()) = 0;
};

class CollidableWithTypesBitVectorVoxelMap
{
public:
  /*!
   * \brief collideWithTypes This does a collision check with 'other' and delivers the voxel meanings that are in collision.
   * This is especially useful when colliding swept volumes and one wants to know, which subvolumes lie in collision.
   * \param map The map to do a collision check with.
   * \param meanings_in_collision The voxel meanings in collision.
   * \param coll_threshold The threshold when a collision is counted. Only valid for probabilistic maps.
   * \param offset The offset in cell coordinates
   * \return The severity of the collision, namely the number of voxels that lie in collision
   */
  virtual size_t collideWithTypes(const voxelmap::BitVectorVoxelMap* map, BitVectorVoxel& types_in_collision, float coll_threshold = 1.0, const Vector3i &offset = Vector3i()) = 0;
};

class CollidableWithBitcheckBitVectorVoxelMap
{
public:
  /*!
   * \brief collideWithBitcheck This does a collision check with 'other' but voxels only collide, if the same bits are set in both.
   * This is especially useful when colliding two swept volumes where the bitvectors represent a point in time.
   * Only available for checks between BitVoxel-Types!
   * \param map The map to do a collision check with.
   * \param margin Allows a more fuzzy check as also n bits around the target bit are checked.
   * \param offset The offset in cell coordinates
   * \return The severity of the collision, namely the number of voxels that lie in collision
   */
  virtual size_t collideWithBitcheck(const voxelmap::BitVectorVoxelMap* map, const u_int8_t margin = 0, const Vector3i &offset = Vector3i()) = 0;
};

// PROB VOXELMAP
class CollidableWithProbVoxelMap
{
public:
  /*!
   * \brief collideWith This does a collision check with 'other'.
   * \param map The map to do a collision check with.
   * \param coll_threshold The threshold when a collision is counted. Only valid for probabilistic maps.
   * \return The severity of the collision, namely the number of voxels that lie in collision
   */
  virtual size_t collideWith(const voxelmap::ProbVoxelMap* map, float coll_threshold = 1.0, const Vector3i &offset = Vector3i()) = 0;
};

class CollidableWithResolutionProbVoxelMap
{
public:
  /*!
   * \brief collideWithResolution This does a collision check with 'other'.
   * \param map The map to do a collision check with.
   * \param coll_threshold The threshold when a collision is counted. Only valid for probabilistic maps.
   * \param resolution_level The resolution used for collision checking. resolution_level = 0 delivers the highest accuracy whereas each increase haves the resolution.
   * \param offset The offset in cell coordinates
   * \return The severity of the collision, namely the number of voxels that lie in collision
   */
  virtual size_t collideWithResolution(const voxelmap::ProbVoxelMap* map, float coll_threshold = 1.0, const uint32_t resolution_level = 0, const Vector3i &offset = Vector3i()) = 0;
};

class CollidableWithTypesProbVoxelMap
{
public:
  /*!
   * \brief collideWithTypes This does a collision check with 'other' and delivers the voxel meanings that are in collision.
   * This is especially useful when colliding swept volumes and one wants to know, which subvolumes lie in collision.
   * \param map The map to do a collision check with.
   * \param meanings_in_collision The voxel meanings in collision.
   * \param coll_threshold The threshold when a collision is counted. Only valid for probabilistic maps.
   * \param offset The offset in cell coordinates
   * \return The severity of the collision, namely the number of voxels that lie in collision
   */
  virtual size_t collideWithTypes(const voxelmap::ProbVoxelMap* map, BitVectorVoxel& types_in_collision, float coll_threshold = 1.0, const Vector3i &offset = Vector3i()) = 0;
};

class CollidableWithBitcheckProbVoxelMap
{
public:
  /*!
   * \brief collideWithBitcheck This does a collision check with 'other' but voxels only collide, if the same bits are set in both.
   * This is especially useful when colliding two swept volumes where the bitvectors represent a point in time.
   * Only available for checks between BitVoxel-Types!
   * \param map The map to do a collision check with.
   * \param margin Allows a more fuzzy check as also n bits around the target bit are checked.
   * \param offset The offset in cell coordinates
   * \return The severity of the collision, namely the number of voxels that lie in collision
   */
  virtual size_t collideWithBitcheck(const voxelmap::ProbVoxelMap* map, const u_int8_t margin = 0, const Vector3i &offset = Vector3i()) = 0;
};

// BITVECTOR VOXELLIST
class CollidableWithBitVectorVoxelList
{
public:
  /*!
   * \brief collideWith This does a collision check with 'other'.
   * \param map The map to do a collision check with.
   * \param coll_threshold The threshold when a collision is counted. Only valid for probabilistic maps.
   * \return The severity of the collision, namely the number of voxels that lie in collision
   */
  virtual size_t collideWith(const voxellist::BitVectorVoxelList* map, float coll_threshold = 1.0, const Vector3i &offset = Vector3i()) = 0;
};

class CollidableWithResolutionBitVectorVoxelList
{
public:
  /*!
   * \brief collideWithResolution This does a collision check with 'other'.
   * \param map The map to do a collision check with.
   * \param coll_threshold The threshold when a collision is counted. Only valid for probabilistic maps.
   * \param resolution_level The resolution used for collision checking. resolution_level = 0 delivers the highest accuracy whereas each increase haves the resolution.
   * \param offset The offset in cell coordinates
   * \return The severity of the collision, namely the number of voxels that lie in collision
   */
  virtual size_t collideWithResolution(const voxellist::BitVectorVoxelList* map, float coll_threshold = 1.0, const uint32_t resolution_level = 0, const Vector3i &offset = Vector3i()) = 0;
};

class CollidableWithTypesBitVectorVoxelList
{
public:
  /*!
   * \brief collideWithTypes This does a collision check with 'other' and delivers the voxel meanings that are in collision.
   * This is especially useful when colliding swept volumes and one wants to know, which subvolumes lie in collision.
   * Only available for checks against BitVoxel-Types!
   * \param map The map to do a collision check with.
   * \param meanings_in_collision The voxel meanings in collision.
   * \param coll_threshold The threshold when a collision is counted. Only valid for probabilistic maps.
   * \param offset The offset in cell coordinates
   * \return The severity of the collision, namely the number of voxels that lie in collision
   */
  virtual size_t collideWithTypes(const voxellist::BitVectorVoxelList* map, BitVectorVoxel& types_in_collision, float coll_threshold = 1.0, const Vector3i &offset = Vector3i()) = 0;
};

class CollidableWithBitcheckBitVectorVoxelList
{
public:
  /*!
   * \brief collideWithBitcheck This does a collision check with 'other' but voxels only collide, if the same bits are set in both.
   * This is especially useful when colliding two swept volumes where the bitvectors represent a point in time.
   * Only available for checks between BitVoxel-Types!
   * \param map The map to do a collision check with.
   * \param margin Allows a more fuzzy check as also n bits around the target bit are checked.
   * \param offset The offset in cell coordinates
   * \return The severity of the collision, namely the number of voxels that lie in collision
   */
  virtual size_t collideWithBitcheck(const voxellist::BitVectorVoxelList* map, const u_int8_t margin = 0, const Vector3i &offset = Vector3i()) = 0;
};

// PROB VOXELLIST
//class CollidableWithProbVoxelList
//{
//public:
//  virtual size_t collideWith(const voxellist:: map, float coll_threshold = 1.0, const Vector3i &offset = Vector3i()) = 0;
//};

//class CollidableWithResolutionProbVoxelList
//{
//public:
//  virtual size_t collideWithResolution(const voxellist:: map, float coll_threshold = 1.0, const uint32_t resolution_level = 0, const Vector3i &offset = Vector3i()) = 0;
//};

//class CollidableWithTypesProbVoxelList
//{
//  virtual size_t collideWithTypes(const voxellist:: map, BitVectorVoxel& types_in_collision, float coll_threshold = 1.0, const Vector3i &offset = Vector3i()) = 0;
//};

//class CollidableWithBitcheckProbVoxelList
//{
//  virtual size_t collideWithBitcheck(const voxellist:: map, const u_int8_t margin = 0, const Vector3i &offset = Vector3i()) = 0;
//};

// BITVECTOR OCTREE
class CollidableWithBitVectorOctree
{
public:
  /*!
   * \brief collideWith This does a collision check with 'other'.
   * \param map The map to do a collision check with.
   * \param coll_threshold The threshold when a collision is counted. Only valid for probabilistic maps.
   * \return The severity of the collision, namely the number of voxels that lie in collision
   */
  virtual size_t collideWith(const NTree::GvlNTreeDet* map, float coll_threshold = 1.0, const Vector3i &offset = Vector3i()) = 0;
};

class CollidableWithResolutionBitVectorOctree
{
public:
  /*!
   * \brief collideWithResolution This does a collision check with 'other'.
   * \param map The map to do a collision check with.
   * \param coll_threshold The threshold when a collision is counted. Only valid for probabilistic maps.
   * \param resolution_level The resolution used for collision checking. resolution_level = 0 delivers the highest accuracy whereas each increase haves the resolution.
   * \param offset The offset in cell coordinates
   * \return The severity of the collision, namely the number of voxels that lie in collision
   */
  virtual size_t collideWithResolution(const NTree::GvlNTreeDet* map, float coll_threshold = 1.0, const uint32_t resolution_level = 0, const Vector3i &offset = Vector3i()) = 0;
};

class CollidableWithTypesBitVectorOctree
{
public:
  /*!
   * \brief collideWithTypes This does a collision check with 'other' and delivers the voxel meanings that are in collision.
   * This is especially useful when colliding swept volumes and one wants to know, which subvolumes lie in collision.
   * Only available for checks against BitVoxel-Types!
   * \param map The map to do a collision check with.
   * \param meanings_in_collision The voxel meanings in collision.
   * \param coll_threshold The threshold when a collision is counted. Only valid for probabilistic maps.
   * \param offset The offset in cell coordinates
   * \return The severity of the collision, namely the number of voxels that lie in collision
   */
  virtual size_t collideWithTypes(const NTree::GvlNTreeDet* map, BitVectorVoxel& types_in_collision, float coll_threshold = 1.0, const Vector3i &offset = Vector3i()) = 0;
};

class CollidableWithBitcheckBitVectorOctree
{
public:
  /*!
   * \brief collideWithBitcheck This does a collision check with 'other' but voxels only collide, if the same bits are set in both.
   * This is especially useful when colliding two swept volumes where the bitvectors represent a point in time.
   * Only available for checks between BitVoxel-Types!
   * \param map The map to do a collision check with.
   * \param margin Allows a more fuzzy check as also n bits around the target bit are checked.
   * \param offset The offset in cell coordinates
   * \return The severity of the collision, namely the number of voxels that lie in collision
   */
  virtual size_t collideWithBitcheck(const NTree::GvlNTreeDet* map, const u_int8_t margin = 0, const Vector3i &offset = Vector3i()) = 0;
};

// PROB OCTREE
class CollidableWithProbOctree
{
public:
  /*!
   * \brief collideWith This does a collision check with 'other'.
   * \param map The map to do a collision check with.
   * \param coll_threshold The threshold when a collision is counted. Only valid for probabilistic maps.
   * \return The severity of the collision, namely the number of voxels that lie in collision
   */
  virtual size_t collideWith(const NTree::GvlNTreeProb* map, float coll_threshold = 1.0, const Vector3i &offset = Vector3i()) = 0;
};

class CollidableWithResolutionProbOctree
{
public:
  /*!
   * \brief collideWithResolution This does a collision check with 'other'.
   * \param map The map to do a collision check with.
   * \param coll_threshold The threshold when a collision is counted. Only valid for probabilistic maps.
   * \param resolution_level The resolution used for collision checking. resolution_level = 0 delivers the highest accuracy whereas each increase haves the resolution.
   * \param offset The offset in cell coordinates
   * \return The severity of the collision, namely the number of voxels that lie in collision
   */
  virtual size_t collideWithResolution(const NTree::GvlNTreeProb* map, float coll_threshold = 1.0, const uint32_t resolution_level = 0, const Vector3i &offset = Vector3i()) = 0;
};

class CollidableWithTypesProbOctree
{
public:
  /*!
   * \brief collideWithTypes This does a collision check with 'other' and delivers the voxel meanings that are in collision.
   * This is especially useful when colliding swept volumes and one wants to know, which subvolumes lie in collision.
   * Only available for checks against BitVoxel-Types!
   * \param map The map to do a collision check with.
   * \param meanings_in_collision The voxel meanings in collision.
   * \param coll_threshold The threshold when a collision is counted. Only valid for probabilistic maps.
   * \param offset The offset in cell coordinates
   * \return The severity of the collision, namely the number of voxels that lie in collision
   */
  virtual size_t collideWithTypes(const NTree::GvlNTreeProb* map, BitVectorVoxel& types_in_collision, float coll_threshold = 1.0, const Vector3i &offset = Vector3i()) = 0;
};

class CollidableWithBitcheckProbOctree
{
public:
  /*!
   * \brief collideWithBitcheck This does a collision check with 'other' but voxels only collide, if the same bits are set in both.
   * This is especially useful when colliding two swept volumes where the bitvectors represent a point in time.
   * Only available for checks between BitVoxel-Types!
   * \param map The map to do a collision check with.
   * \param margin Allows a more fuzzy check as also n bits around the target bit are checked.
   * \param offset The offset in cell coordinates
   * \return The severity of the collision, namely the number of voxels that lie in collision
   */
  virtual size_t collideWithBitcheck(const NTree::GvlNTreeProb* map, const u_int8_t margin = 0, const Vector3i &offset = Vector3i()) = 0;
};

// BITVECTOR MORTON VOXELLIST
class CollidableWithBitVectorMortonVoxelList
{
public:
  /*!
   * \brief collideWith This does a collision check with 'other'.
   * \param map The map to do a collision check with.
   * \param coll_threshold The threshold when a collision is counted. Only valid for probabilistic maps.
   * \return The severity of the collision, namely the number of voxels that lie in collision
   */
  virtual size_t collideWith(const voxellist::BitVectorMortonVoxelList* map, float coll_threshold = 1.0, const Vector3i &offset = Vector3i()) = 0;
};

class CollidableWithResolutionBitVectorMortonVoxelList
{
public:
  /*!
   * \brief collideWithResolution This does a collision check with 'other'.
   * \param map The map to do a collision check with.
   * \param coll_threshold The threshold when a collision is counted. Only valid for probabilistic maps.
   * \param resolution_level The resolution used for collision checking. resolution_level = 0 delivers the highest accuracy whereas each increase haves the resolution.
   * \param offset The offset in cell coordinates
   * \return The severity of the collision, namely the number of voxels that lie in collision
   */
  virtual size_t collideWithResolution(const voxellist::BitVectorMortonVoxelList* map, float coll_threshold = 1.0, const uint32_t resolution_level = 0, const Vector3i &offset = Vector3i()) = 0;
};

class CollidableWithTypesBitVectorMortonVoxelList
{
public:
  /*!
   * \brief collideWithTypes This does a collision check with 'other' and delivers the voxel meanings that are in collision.
   * This is especially useful when colliding swept volumes and one wants to know, which subvolumes lie in collision.
   * Only available for checks against BitVoxel-Types!
   * \param map The map to do a collision check with.
   * \param meanings_in_collision The voxel meanings in collision.
   * \param coll_threshold The threshold when a collision is counted. Only valid for probabilistic maps.
   * \param offset The offset in cell coordinates
   * \return The severity of the collision, namely the number of voxels that lie in collision
   */
  virtual size_t collideWithTypes(const voxellist::BitVectorMortonVoxelList* map, BitVectorVoxel& types_in_collision, float coll_threshold = 1.0, const Vector3i &offset = Vector3i()) = 0;
};

class CollidableWithBitcheckBitVectorMortonVoxelList
{
public:
  /*!
   * \brief collideWithBitcheck This does a collision check with 'other' but voxels only collide, if the same bits are set in both.
   * This is especially useful when colliding two swept volumes where the bitvectors represent a point in time.
   * Only available for checks between BitVoxel-Types!
   * \param map The map to do a collision check with.
   * \param margin Allows a more fuzzy check as also n bits around the target bit are checked.
   * \param offset The offset in cell coordinates
   * \return The severity of the collision, namely the number of voxels that lie in collision
   */
  virtual size_t collideWithBitcheck(const voxellist::BitVectorMortonVoxelList* map, const u_int8_t margin = 0, const Vector3i &offset = Vector3i()) = 0;
};

// PROB MORTON VOXELLIST
//class CollidableWithProbMortonVoxelList
//{
//public:
//  virtual size_t collideWith(const voxellist:: map, float coll_threshold = 1.0, const Vector3i &offset = Vector3i()) = 0;
//};

//class CollidableWithResolutionProbMortonVoxelList
//{
//public:
//  virtual size_t collideWithResolution(const voxellist:: map, float coll_threshold = 1.0, const uint32_t resolution_level = 0, const Vector3i &offset = Vector3i()) = 0;
//};

//class CollidableWithTypesProbMortonVoxelList
//{
//  virtual size_t collideWithTypes(const voxellist:: map, BitVectorVoxel& types_in_collision, float coll_threshold = 1.0, const Vector3i &offset = Vector3i()) = 0;
//};

//class CollidableWithBitcheckProbMortonVoxelList
//{
//  virtual size_t collideWithBitcheck(const voxellist:: map, const u_int8_t margin = 0, const Vector3i &offset = Vector3i()) = 0;
//};

} //end namespace gpu_voxels

#endif // GPU_VOXELS_HELPERS_COLLISIONINTERFACES_H_INCLUDED
