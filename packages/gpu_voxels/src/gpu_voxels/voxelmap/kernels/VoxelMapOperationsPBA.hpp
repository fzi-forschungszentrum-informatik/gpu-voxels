// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
/*
===============================================================================

Copyright (c) 2010, School of Computing, National University of Singapore.
All rights reserved.

Project homepage: http://www.comp.nus.edu.sg/~tants/pba.html

If you use PBA and you like it or have comments on its usefulness etc., we
would love to hear from you at <tants@comp.nus.edu.sg>. You may share with us
your experience and any possibilities that we may improve the work/code.

===============================================================================

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution.

Neither the name of the National University of University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission from the National University of Singapore.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.

*/
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \brief   Original PBA implementation by Cao Thanh Tung. Modified by others.
 * \author  Cao Thanh Tung
 * \author  Christian Jülg
 * \date    2010-2016
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_PLANNING_GPU_KERNELS_VOXELMAP_OPERATIONS_PBA_HPP_INCLUDED
#define ICL_PLANNING_GPU_KERNELS_VOXELMAP_OPERATIONS_PBA_HPP_INCLUDED

#include "VoxelMapOperations.hpp"

namespace gpu_voxels {
namespace voxelmap {

template<typename InputIterator1, typename InputIterator2>
__global__
void kernelPBAphase1FloodZ(InputIterator1 input, InputIterator2 output, const Vector3ui dims, int band_size) {
    if (dims.x < blockDim.x) {
      if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("ERROR: kernelPBAphase1FloodZ: dims.x too small, must be at least blockDim.x\n");
      }
    }

    const int tx = blockIdx.x * blockDim.x + threadIdx.x; //iterates 0..dim.x
    const int ty = blockIdx.y * blockDim.y + threadIdx.y; //iterates 0..dim.y
    const int tz_bottom = blockIdx.z * band_size; //blockIdx.z iterates 0..m1

    const int layer_size = dims.x * dims.y;

    //flood forward
    uint3 obstacle = make_uint3(PBA_UNINITIALISED_COORD, PBA_UNINITIALISED_COORD, PBA_UNINITIALISED_COORD);

    // linear index
    const int band_base_id = (tz_bottom * layer_size) + (ty * dims.x) + tx;

    //TODO: check whether distance value is ever used to determine obstacle location before phase3 writes distances for every voxel?
    //optimize: if check says distances are not used, remove all distance related code, when SoA is used

    for (int i = 0; i < band_size; i++) {
        int voxel_id = band_base_id + (i * layer_size);
        uint3 current = (DistanceVoxel)input[voxel_id];
        uint current_z = current.z;

        if (current_z != PBA_UNINITIALISED_COORD) {
          obstacle = current;
        }

        //optimise: don't save distances -> after pba finishes, do distance transform?
        output[voxel_id] = DistanceVoxel(obstacle); //distance always >= 0
    }

    //flood backward
    obstacle = make_uint3(PBA_UNINITIALISED_COORD, PBA_UNINITIALISED_COORD, PBA_UNINITIALISED_COORD);

    // linear index
    const int tz_top = tz_bottom + band_size - 1;
    const int band_top_id = band_base_id + ((band_size - 1) * layer_size);

    int dist_prev_obstacle, dist_next_obstacle;
    for (int i = 0; i < band_size; i++) {
        int voxel_id = band_top_id - (i * layer_size); //walk left

        //TODO: rethink update mechanism; are there unnecessary steps?
        dist_prev_obstacle = abs(((int)obstacle.z) - (tz_top - i)); //obstacle.z >= (tz_top-i)

        //read results from forward flood
        uint3 next_obstacle = (DistanceVoxel)output[voxel_id];
        dist_next_obstacle = abs((tz_top - i) - (int)next_obstacle.z); //next_obstacle.z == (tz_top-i) unless next_obstacle is PBA_UNINIT

        if (dist_next_obstacle <= dist_prev_obstacle) { // prefer lower coordinate if distance equal
            obstacle = next_obstacle;
        }

        output[voxel_id] = DistanceVoxel(obstacle);
    }
}

// optimise: could be done per line instead of per band. but per band prob. faster if no propagation over many bands; would divide work by m1 at most
// optimise: is shared memory large enough for Y*m1*2 Voxels (2 per band)? if all bands in a line are on same block, shared memory could be efficient? is running time relevant?
template<typename InputIterator1, typename InputIterator2>
__global__
void kernelPBAphase1PropagateInterband(InputIterator1 input, InputIterator2 output, const Vector3ui dims, int band_size) {
  // for all bands (one thread per band):
  // "combine" first pixel with bands on left
  // "combine" last pixel with band on right

  const int tx = blockIdx.x * blockDim.x + threadIdx.x; //iterates 0..dim.x
  const int ty = blockIdx.y * blockDim.y + threadIdx.y; //iterates 0..dim.y

  const int tz_bottom = blockIdx.z * band_size; //blockIdx.z iterates 0..m1
  const int tz_top = tz_bottom + band_size - 1;

  const int layer_size = dims.x * dims.y;

  // read first band voxel, look at last voxels on left and update if new closest obstacle found:
  const int band_bottom_voxel_id = (tz_bottom * layer_size) + (ty * dims.x) + tx;
  const int band_top_voxel_id =  band_bottom_voxel_id + ((band_size - 1) * layer_size);

  // linear index
  uint3 current = (DistanceVoxel)input[band_bottom_voxel_id];
  int cur_dist = abs(((int)current.z) - tz_bottom); // current must be within band

  for (int offset = band_size; offset <= tz_top; offset += band_size) { // move band by band backward
    //look at last voxel of bands that come before
    current = (DistanceVoxel)input[band_top_voxel_id - (offset * layer_size)];

    if (current.z != PBA_UNINITIALISED_COORD) { //is initialised
      int new_dist = abs(((int)current.z) - tz_bottom);

      if (new_dist < cur_dist)
          output[band_bottom_voxel_id] = DistanceVoxel(current);

      break; // further obstacles can't be closer than newDist
    }
  }

  // read last band voxel, look at first voxels on right and update last band voxel if lower distance found
  current = (DistanceVoxel)input[band_top_voxel_id];
  cur_dist = abs(((int)current.z) - tz_top); //current must be within band or uninitialised

  for (int offset = band_size; tz_top + offset < dims.z; offset += band_size) { // move band by band onward
    //look at first voxel of bands that come after
    current = (DistanceVoxel)input[band_bottom_voxel_id + (offset * layer_size)];

    if (current.z != PBA_UNINITIALISED_COORD) { //is initialised
      int new_dist = abs(((int)current.z) - tz_top);

      if (new_dist < cur_dist)
          output[band_top_voxel_id] = DistanceVoxel(current);

      break; // further obstacles can't be closer than newDist
    }
  }
}

/**
 * //buffer to b; a is Links (top,bottom), b is Color (voxel)
 */
template<typename InputIterator1, typename InputIterator2>
__global__
void kernelPBAphase1Update(InputIterator1 input, InputIterator2 output, const Vector3ui dims, int band_size)
{
  //optimise: in phase 1 all obstacle information is "z only". eliminate setobstacle, just write z? update phase1, 2

  const int tx = blockIdx.x * blockDim.x + threadIdx.x; //iterates 0..dim.x
  const int ty = blockIdx.y * blockDim.y + threadIdx.y; //iterates 0..dim.y

  const int tz_bottom = blockIdx.z * band_size; //blockIdx.z iterates 0..m1

  const int layer_size = dims.x * dims.y;

  // read first band voxel, look at last voxels on left:
  const int band_bottom_voxel_id = (tz_bottom * layer_size) + (ty * dims.x) + tx;
  const int band_top_voxel_id =  band_bottom_voxel_id + ((band_size - 1) * layer_size);

  //left and right are in output; propagateInterband modfied output only
  //optmise: maybe copy all valid information from in to out in interband step?
  uint3 left = (DistanceVoxel)input[band_bottom_voxel_id];

  uint3 right = (DistanceVoxel)input[band_top_voxel_id];

  int new_dist, min_dist;

  for (int i = 0; i < band_size; i++) { // optimise: if input = output, start from 1, stop at item before last
    int id = band_bottom_voxel_id + (i * layer_size);

    //current is read from output as input only contains information in first&last voxel of every band
    uint3 current = (DistanceVoxel)output[id];
    bool new_minimum_found = false;

    min_dist = abs(((int)current.z) - (tz_bottom + i)); //distance of current obstacle to current pos

    new_dist = abs(((int)left.z) - (tz_bottom + i));
    if (new_dist < min_dist) {
      current = left; //these voxels are within same row; important: difference is positive
      min_dist = new_dist;
      new_minimum_found = true;
    }

    new_dist = abs(((int)right.z) - (tz_bottom + i));
    if (new_dist < min_dist) {
      current = right; //these voxels are within same row; important: difference is positive
      min_dist = new_dist;
      new_minimum_found = true;
    }

    if (new_minimum_found) {
      output[id] = DistanceVoxel(current);
    }
  }
}

/**
 * @brief find intersection of perpendicular bisector of a<->b and the y-column in row x0
 * input: obstacles a(x1/y1), b(x2/y2) where y1 < y2 and column x0
 *
 * only used in 2D-PBA
 */
__device__
float devicePBAintersection2D(int x1, int y1, int x2, int y2, int x0)
{
  // xM and yM are coordinates of a<->b meeting its perpendicular bisector
  float xM = float(x1 + x2) / 2.0f;
  float yM = float(y1 + y2) / 2.0f;
  float nx = x2 - x1;
  float ny = y2 - y1;

  // use known inclination and known point (xM/yM) to get equation
  // return intersection of perpendicular bisector and column x0
  return yM + nx * (xM - x0) / ny;
}

/**
 * @brief find intersection of perpendicular bisector of a<->b and the y-column in row x0 and z-layer z0
 * input: obstacles a(x1/y1), b(x2/y2) where y1 < y2 and column x0
 */
__device__ float devicePBAintersection3DY(int x1, int y1, int z1, int x2, int y2, int z2, int x0, int z0)
{
  // xM/yM/zM are coordinates of a<->b meeting its perpendicular bisecting plane
  float xM = (x1 + x2) / 2.0f;
  float yM = (y1 + y2) / 2.0f;
  float zM = (z1 + z2) / 2.0f;
  float nx = x2 - x1;
  float ny = y2 - y1;
  float nz = z2 - z1;

  return yM + (nx * (xM - x0) + nz * (zM - z0)) / ny;
}

/**
 * PBA phase 2
 *
 * input is read-only, stack is read/write
 *
 * voxels in the stack array store backpointers to previous obstacle information in the band using y-coordinate
 *
 * if no previous obstacle exists, y value is set to PBA_UNINITIALISED_COORD
 *
 * if last voxel of band does not contain obstacle information, it becomes a TAIL pointer and has x and z values of PBA_UNINITIALISED_COORD
 */
template<typename InputIterator1, typename InputIterator2>
__global__
void kernelPBAphase2ProximateBackpointers(InputIterator1 input_map, InputIterator2 stack, const Vector3ui dims, int band_size){
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int band_idx = blockIdx.y * blockDim.y + threadIdx.y; //blockDim.y should be 1
  int ty_bottom = band_idx * band_size;
  int tz = blockIdx.z * blockDim.z + threadIdx.z;   //blockDim.z should be 1

  int last_y = PBA_UNINITIALISED_COORD;

  //stack_top is top of stack; stack_second is "item below top"
  uint3 stack_top;
  uint3 stack_second;

  stack_top.y = PBA_UNINITIALISED_COORD;
  stack_second.y = PBA_UNINITIALISED_COORD;

  uint3 current = make_uint3(PBA_UNINITIALISED_COORD, PBA_UNINITIALISED_COORD, PBA_UNINITIALISED_COORD);
  float intersection1, intersection2;

  const int row_size = dims.x;
  const int layer_size = dims.x * dims.y;
  const int band_bottom_voxel_id = (tz * layer_size) + (ty_bottom * row_size) + tx;

  //for all voxels in band:
  for (int i = 0; i < band_size; i++) {
    uint id = band_bottom_voxel_id + (i * row_size);
    int ty = ty_bottom + i;

    current = (DistanceVoxel)input_map[id];

    //skip all voxels without obstacle info (empty z-column in phase1)
    if (current.x != PBA_UNINITIALISED_COORD) { //has obstacle information; always true if there were obstacles anywhere within z-column
      // because of this check, empty z-columns from phase1Update remain blank during kernelPBAphase2ProximateBackpointers

      while (stack_top.y != PBA_UNINITIALISED_COORD) { //stack_top and stack_second initialised; if not, load current to stack_top, but stack_top.y still uninitialised
        intersection1 = devicePBAintersection3DY(stack_second.x, stack_top.y, stack_second.z, stack_top.x, last_y, stack_top.z, tx, tz);
        intersection2 = devicePBAintersection3DY(stack_top.x, last_y, stack_top.z, current.x, ty, current.z, tx, tz);

        if (intersection1 < intersection2)
          break; //don't eliminate stack_top from band1

        //else: forget stack_top (coordinates: stack_top.x, last_y, stack_top.z); it's dominated by stack_second and current

//        //TODO remove
//        printf("proximateBackpointers: omitting point %u %u %u proximateBackpointers: previous point %u %u %u proximateBackpointers: next point %u %u %u\nintersects %f %f, i: %i, thread: %d %d %d\n",
//               stack_top.x, last_y, stack_top.z,
//               stack_second.x, stack_top.y, stack_second.z,
//               current.x, ty, current.z,
//               intersection1, intersection2, i,
//               tx, ty, tz);
//        //TODO remove

        // pop stack_top: top=second, second = *(second.backpointer)
        last_y = stack_top.y;
        stack_top = stack_second;

        if (stack_top.y != PBA_UNINITIALISED_COORD) {
          //load voxel indicated by stack_top.y into stack_second

//          voxel = stack[ (tz * layer_size) + (stack_top.y * row_size) + tx ];
//          stack_second = voxel;
          stack_second = (DistanceVoxel) stack[ (tz * layer_size) + (stack_top.y * row_size) + tx ];

          //will reenter while loop! same condition regarding stack_top.y
          //old value of stack_second has been saved in stack_top
        }
      }

      //last_y contains last obstacle encountered (in current)

      //voxels are read into current; move from current through stack_top to stack_second;
      //whenever current contains an obstacle, its x and z will be written to stack at the same index; the y value will be last_y
      stack_second = stack_top;
      stack_top = make_uint3(current.x, last_y, current.z);
      last_y = ty;

      stack[ id ] = DistanceVoxel(stack_top);
    }
  }

  if (current.x == PBA_UNINITIALISED_COORD) {
    //last voxel in band is not an obstacle. write tail pointer to last band voxel
    stack[ band_bottom_voxel_id + ((band_size - 1 ) * row_size) ] = DistanceVoxel(Vector3ui(PBA_UNINITIALISED_COORD, last_y, PBA_UNINITIALISED_COORD));
  }

  // result:
  // stack contains (phase1_obstacle.x, last_y, phase1_obstacle.z) in all the positions that had obstacle information in input
  // last voxel in stack will contain tail pointer, if it had no obstacle info
  // all other previously empty voxels will remain untouched? weren't touched in phase1?
}

template<typename InputIterator1, typename InputIterator2>
__global__
void kernelPBAphase2CreateForwardPointers(InputIterator1 list, InputIterator2 forward_ptrs, const Vector3ui dims, int band_size){
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int band_idx = blockIdx.y * blockDim.y + threadIdx.y; //blockDim.y should be 1
  int ty_top = ((band_idx + 1) * band_size) - 1;
  int tz = blockIdx.z * blockDim.z + threadIdx.z;   //blockDim.z should be 1

  pba_fw_ptr_t last_y = PBA_UNINITIALISED_FORW_PTR;
  pba_fw_ptr_t next_y;

  const int row_size = dims.x;
  const int layer_size = dims.x * dims.y;
  const int band_top_voxel_id = (tz * layer_size) + (ty_top * row_size) + tx;

  {
    uint3 current = (DistanceVoxel)list[band_top_voxel_id];
    // Get the tail pointer
    if (current.x == PBA_UNINITIALISED_COORD) {
      next_y = current.y;  //location of next obstacle (in -y Dir), or PBA_UNINIT
    } else {
      next_y = ty_top; // last band voxel is not a pointer but has obstacle information itself
    }
  }

  uint id = band_top_voxel_id;
  for (int i = 0; i < band_size; i++) {
    int ty = ty_top - i;
    id = band_top_voxel_id - (i * row_size);

    if (ty == next_y) {
      forward_ptrs[id] = last_y; //optimise: in SoA case, write only last_y; need to differentiate normal forward pointer and HEAD element; now distance<0 designates HEAD; could use bit 1<<31 if distance field is eliminated
      next_y = ((DistanceVoxel)list[id]).getObstacle().y;
      last_y = ty;
    }
  }

  // Store the pointer to the head at the first pixel of this band
  // could be -PBA_UNINITIALISED_COORD if current.y is PBA_UNINITIALISED_COORD (i.e. band is empty)
  if (last_y != ty_top - (band_size - 1)) { // if ty_bottom element had no obstacle info and was never back-pointed to? first obstacle back-points to PBA_UNINITIALISED_COORD

    //this should be a proper head pointer; value is negative to indicate head-pointer status; value could be -PBA_UNINITIALISED_COORD if band empty
    forward_ptrs[band_top_voxel_id - ((band_size - 1) * row_size)] = -last_y;
  }
}

/**
 * @brief merge two bands into one
 *
 * eliminates points in the "upper" (previous, geometrically higher, has lower coordinates, band1) band that area dominated by obstacles in "lower" (next, band2) band;
 * previous band is traversed using back-pointers in stack
 * next band is traversed using forward pointers in output_forward (format: forward_ptrs[id] = DistanceVoxel(Vector3ui(), last_y); )
 *
 * process is done as soon as two obstacles from next band have been merged
 *
 * stack:
 *  represents a singly-linked list within an array;
 *  last element is a TAIL pointer if no obstacle stored there;
 *  array elements that are not pointed to by a (backward or forward) pointer will not contribute to the final distance map
 *  DistanceVoxel distance value is never read from stack[] elements;
 *   their distance should always be PBA_UNINITIALISED_COORD
 *  obstacle.y values represent backpointers
 *  the TAIL element is indicated by obstacle.x == PBA_UNINITIALISED_COORD
 *
 * forward_ptrs:
 *  DistanceVoxels.distance represents forward pointers
 *  if distance is negative, the DistanceVoxel denotes the HEAD pointer element
 *
 * during the merge obstacles of band1 will be eliminated if they are dominated by previous band1 and following band2 obstacles
 * to merge the two bands, relevant HEAD, TAIL, forward&backpointers need to be updated
 */
template<typename InputIterator1, typename InputIterator2>
__global__
void kernelPBAphase2MergeBands(InputIterator1 stack, InputIterator2 forward_ptrs, const Vector3ui dims, int band_size){
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int band1_idx = (blockIdx.y * blockDim.y + threadIdx.y) * 2;  //blockDim.y should be 1
  int band2_idx = band1_idx + 1;
  int tz = blockIdx.z * blockDim.z + threadIdx.z;   //blockDim.z should be 1

  const int row_size = dims.x;
  const int layer_size = dims.x * dims.y;

  //init to ty of HEAD of band 2 / TAIL of band 1
  pba_fw_ptr_t y_next_band2 = band2_idx * band_size;
  pba_fw_ptr_t y_last_band1 = y_next_band2 - 1;
  int id_next_band2;

  uint3 stack_second, stack_top, voxel_next_band2;
  float intersection1, intersection2;

  // Band 1, get the two last items
  stack_top = (DistanceVoxel)stack[ (tz * layer_size) + (y_last_band1 * row_size) + tx ];


  if (stack_top.x == PBA_UNINITIALISED_COORD) {     // -> tail pointer or band1 empty
    y_last_band1 = stack_top.y;

    if (y_last_band1 != PBA_UNINITIALISED_FORW_PTR) {  //only false if all slices in band empty?
      stack_top = (DistanceVoxel)stack[ (tz * layer_size) + (y_last_band1 * row_size) + tx ];
    }
  }

  if (stack_top.y != PBA_UNINITIALISED_COORD) {
    stack_second = (DistanceVoxel)stack[ (tz * layer_size) + (stack_top.y * row_size) + tx ];
  }

  // Band 2, get the first item
  id_next_band2 = (tz * layer_size) + (y_next_band2 * row_size) + tx;
  {
    // format was forward_ptrs[id] = DistanceVoxel(Vector3ui(), last);
    pba_fw_ptr_t head_y_band2 = forward_ptrs[ id_next_band2 ]; //optimise: use SoA, get just the int

    if (head_y_band2 < 0) {		// -> is head pointer
      // what does the next band voxel look like after phase2forward? ->  pointer value is negative to indicate head-pointer status; value could be -PBA_UNINITIALISED_COORD if band empty
      // this is an invalid coordinate if head_y_band2 was -PBA_UNINITIALISED
      y_next_band2 = -head_y_band2;  // negate to get forward-pointer
      id_next_band2 = (tz * layer_size) + (y_next_band2 * row_size) + tx;
    }
    //else: next stays pointed at first element of band2, which contains obstacle info
  }

  if (y_next_band2 != PBA_UNINITIALISED_FORW_PTR) {  // -> there is a next obstacle voxel in band2
    voxel_next_band2 = (DistanceVoxel)stack[ id_next_band2 ];
  }
  //else: next_band2 is PBA_UNINITIALISED, no obstacles in band2 after next_band2

  int band2_kept = 0;

  // until top 2 elements of stack are from band2 (-> happens when no more elements of band1 get dominated)
  // or reached end of band2
  while (band2_kept < 2 && y_next_band2 != PBA_UNINITIALISED_FORW_PTR) {
    while (stack_top.y != PBA_UNINITIALISED_COORD) { //while at east two elements left in band1 (i.e. stack_top and stack_second have obstacle info and have not yet been dominated)
      intersection1 = devicePBAintersection3DY(stack_second.x, stack_top.y, stack_second.z, stack_top.x, y_last_band1, stack_top.z, tx, tz);
      intersection2 = devicePBAintersection3DY(stack_top.x, y_last_band1, stack_top.z, voxel_next_band2.x, y_next_band2, voxel_next_band2.z, tx, tz);

      if (intersection1 < intersection2)
        break; //don't eliminate stack_top from band1

      //eliminate stack_top from band1
      y_last_band1 = stack_top.y;
      stack_top = stack_second;
      band2_kept--;

      if (stack_top.y != PBA_UNINITIALISED_COORD) { //load element before stack_second, if valid
        stack_second = (DistanceVoxel)stack[ (tz * layer_size) + (stack_top.y * row_size) + tx ];
      }
    }

    // Update pointers to link voxel_next_band2 to the stack

    // modify back_pointer in voxel_next_band2 from band2 to point to tail of band1 / stack
    stack[id_next_band2] = DistanceVoxel(Vector3ui(voxel_next_band2.x, y_last_band1, voxel_next_band2.z));


    // fix forward pointer to next_band2
    if (y_last_band1 != PBA_UNINITIALISED_FORW_PTR) { //band1 has more elements
      forward_ptrs[ (tz * layer_size) + (y_last_band1 * row_size) + tx ]
              = y_next_band2; //forward format, not a HEAD -> positive; //optimise
    }

    band2_kept = max(1, band2_kept + 1); // we keep one more from band2

    // advance voxel_next_band2 forward
    // old current becomes stack_top
    stack_second = stack_top; stack_top = make_uint3(voxel_next_band2.x, y_last_band1, voxel_next_band2.z);
    y_last_band1 = y_next_band2;
    y_next_band2 = forward_ptrs[ id_next_band2 ]; //optimise SoA
    if (y_next_band2 < 0) { //can happen
        y_next_band2 = - y_next_band2;
    }

    // load next band2 element
    if (y_next_band2 != PBA_UNINITIALISED_FORW_PTR) { //more elements in band2?
      id_next_band2 = (tz * layer_size) + (y_next_band2 * row_size) + tx;
      voxel_next_band2 = (DistanceVoxel)stack[ id_next_band2 ];
    }
  }

  // Update the head pointer

  const pba_fw_ptr_t ty_start_band1 = band1_idx * band_size; //point to start of band1
  const pba_fw_ptr_t ty_start_band2 = band2_idx * band_size;  //point to start of band2

  // ensure CreateForward will set empty band's HEAD=-PBA_UNINIT_FORW_PTR
  if (forward_ptrs[ (tz * layer_size) + (ty_start_band1 * row_size) + tx ] == -PBA_UNINITIALISED_FORW_PTR) { //if band1 empty
    //set band1 head pointer to head of band2; value must be negative to denote this is a HEAD
    pba_fw_ptr_t dist_start_band2 = forward_ptrs[ (tz * layer_size) + (ty_start_band2 * row_size) + tx ];
    if (dist_start_band2 < 0) {
      // first voxel in band2 was a HEAD
      forward_ptrs[ (tz * layer_size) + (ty_start_band1 * row_size) + tx ] = dist_start_band2; //dist is negative
    } else {
      // first voxel in band2 was NOT a HEAD, store ty_start_band2 as negative number, denoting HEAD status
      forward_ptrs[ (tz * layer_size) + (ty_start_band1 * row_size) + tx ] = -ty_start_band2;
    }
  }

  // Update the tail pointer
  const int ty_end_band1 = ty_start_band1 + band_size - 1; //point to end of band1
  const int ty_end_band2 = ty_start_band2 + band_size - 1; //point to end of band2

  // points to last voxel of band2
  const int id_end_band2 = (tz * layer_size) + (ty_end_band2 * row_size) + tx;
  uint3 voxel_end_band2 = (DistanceVoxel)stack[ id_end_band2 ];

  if (voxel_end_band2.x == PBA_UNINITIALISED_COORD && voxel_end_band2.y == PBA_UNINITIALISED_COORD) { //band2 empty?
    stack_second = (DistanceVoxel)stack[ (tz * layer_size) + (ty_end_band1 * row_size) + tx ];

    if (stack_second.x == PBA_UNINITIALISED_COORD) // if band1 has non-obstacle tail element, use its y-information
      voxel_end_band2.y = stack_second.y;
    else // else point to last element of band1
      voxel_end_band2.y = ty_end_band1;

    //write to last voxel of band2
    stack[id_end_band2] = DistanceVoxel(voxel_end_band2);
  }
}

/**
 * input is "stack" containing P_i; output is complete distance map
 * stack has obstacle x, z info and "y-backpointers" to previous obstacle
 *
 * m3_block_size = dim3(PBA_M3_BLOCKX (==16), m3); // M3_BLOCKX is the number of bands; m3 is the number of threads within each band
 */
template<typename InputIterator1, typename InputIterator2>
__global__
void kernelPBAphase3Distances(InputIterator1 stack, InputIterator2 distance_map, const Vector3ui dims) {
  //TODO: replace all accesses to stack with cuda texture object references; texture<int>

  __shared__ uint3 s_stack_second[PBA_M3_BLOCKX];
  __shared__ uint3 s_stack_top[PBA_M3_BLOCKX];
  __shared__ uint s_last_y[PBA_M3_BLOCKX];
  __shared__ float s_intersection[PBA_M3_BLOCKX];

  const int band_id = threadIdx.y; // in 0..(m3-1)
  const int band_count = blockDim.y; // == m3

  const int row_size = dims.x;
  const int layer_size = dims.x * dims.y;

  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int tz = blockIdx.z;

  uint3 stack_second, stack_top;
  int last_y;
  float intersection;

  /*
   * forward pointers are not used at all
   *
   * how is shared memory used by threads in same band?
   *    every group of m3 threads in same column will read same shared value and check it for validity
   */

  // if thread is last within column, load shared values
  if (band_id == band_count - 1) {
    last_y = dims.y - 1;

    stack_top = (DistanceVoxel)stack[ (tz * layer_size) + (last_y * row_size) + tx ];

    if (stack_top.x == PBA_UNINITIALISED_COORD) {     // -> tail pointer
      last_y = stack_top.y;

      if (last_y != PBA_UNINITIALISED_COORD) { //else: no obstacle in image?
        stack_top = (DistanceVoxel)stack[ (tz * layer_size) + (last_y * row_size) + tx ];
      }
    }

    if (stack_top.y != PBA_UNINITIALISED_COORD) {
      stack_second = (DistanceVoxel)stack[ (tz * layer_size) + (stack_top.y * row_size) + tx ];

      // perpendicular bisector of obstacles stack1 and stack2 intersects y-column at (tx, intersection, tz)
      intersection = devicePBAintersection3DY(stack_second.x, stack_top.y, stack_second.z, stack_top.x, last_y, stack_top.z, tx, tz);
    }

    s_stack_second[threadIdx.x] = stack_second;
    s_stack_top[threadIdx.x] = stack_top;
    s_last_y[threadIdx.x] = last_y;
    s_intersection[threadIdx.x] = intersection;
  }

  __syncthreads();

  for (int ty = dims.y - 1 - band_id; ty >= 0; ty -= band_count) { //band_id is 0..(m3-1); band_count is m3
    //all m3 threads read same shared value!; the larger m3, the longer the while loop will need to run, causing lots of divergence
    //optimise: find optimal m3; minimise divergence?
    stack_second = s_stack_second[threadIdx.x];
    stack_top = s_stack_top[threadIdx.x];
    last_y = s_last_y[threadIdx.x];
    intersection = s_intersection[threadIdx.x];

    //performance: the fewer different voronoi regions intersect the part of the y-column processed by this group of m3 threads, the less iterations in this while loop
    while (stack_top.y != PBA_UNINITIALISED_COORD) { // update intersection, stack_top/second until ty > intersection
      if (ty > intersection)
        break; //don't update intersection unless ty <= intersection

      // how to treat ty == intersection? want to save smaller coordinates!
      // what if ty == 0 == intersection? -> then stack_second.y is invalid and no new value will get loaded

      // else: forget stack_top (top of stack); ty has "crossed" the intersector; also update intersection
      //      top=second, second = *(second.backpointer); intersection = ...

      last_y = stack_top.y;
      stack_top = stack_second;

      if (stack_top.y != PBA_UNINITIALISED_COORD) {
        stack_second = (DistanceVoxel)stack[ (tz * layer_size) + (stack_top.y * row_size) + tx ];
        intersection = devicePBAintersection3DY(stack_second.x, stack_top.y, stack_second.z, stack_top.x, last_y, stack_top.z, tx, tz);
      }
    }

    __syncthreads();

    // coalesced global write
    distance_map[(tz * layer_size) + (ty * row_size) + tx] = DistanceVoxel(Vector3ui(stack_top.x, last_y, stack_top.z));

    // thread with max band_id has minmal ty -> most accurate info: update shared values
    if (band_id == band_count - 1) {
      s_stack_second[threadIdx.x] = stack_second; s_stack_top[threadIdx.x] = stack_top; s_last_y[threadIdx.x] = last_y; s_intersection[threadIdx.x] = intersection;
    }

    __syncthreads();
  }
}

/**
 * input is "stack" containing P_i; output is complete distance map
 * stack has obstacle x, z info and "y-backpointers" to previous obstacle
 *
 * m3_block_size = dim3(PBA_M3_BLOCKX (==16), m3); // M3_BLOCKX is the number of bands; m3 is the number of threads within each band
 *
 * partial template specialisation for texture input!
 */
template<typename InputIterator2>
__global__
void kernelPBAphase3Distances(cudaTextureObject_t stack, InputIterator2 distance_map, const Vector3ui dims) {
  //TODO: replace all accesses to stack with cuda texture object references; texture<int>

  __shared__ uint3 s_stack_second[PBA_M3_BLOCKX];
  __shared__ uint3 s_stack_top[PBA_M3_BLOCKX];
  __shared__ uint s_last_y[PBA_M3_BLOCKX];
  __shared__ float s_intersection[PBA_M3_BLOCKX];

  const int band_id = threadIdx.y; // in 0..(m3-1)
  const int band_count = blockDim.y; // == m3

  const int row_size = dims.x;
  const int layer_size = dims.x * dims.y;

  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int tz = blockIdx.z;

  uint3 stack_second, stack_top;
  int last_y;
  float intersection;

  /*
   * forward pointers are not used at all
   *
   * how is shared memory used by threads in same band?
   *    every group of m3 threads in same column will read same shared value and check it for validity
   */

  // if thread is last within column, load shared values
  if (band_id == band_count - 1) {
    last_y = dims.y - 1;

    //TODO: introduce { int tmp = tex1Dfetch(stack, i); stack_top = (DistanceVoxel)tmp); ???
    stack_top = (DistanceVoxel)(tex1Dfetch<int>(stack, (tz * layer_size) + (last_y * row_size) + tx ));
//    stack_top = (DistanceVoxel)(tex1Dfetch(stack, (tz * layer_size) + (last_y * row_size) + tx ));

    if (stack_top.x == PBA_UNINITIALISED_COORD) {     // -> tail pointer
      last_y = stack_top.y;

      if (last_y != PBA_UNINITIALISED_COORD) { //else: no obstacle in image?
        stack_top = (DistanceVoxel)(tex1Dfetch<int>(stack, (tz * layer_size) + (last_y * row_size) + tx ));
      }
    }

    if (stack_top.y != PBA_UNINITIALISED_COORD) {
      stack_second = (DistanceVoxel)(tex1Dfetch<int>(stack, (tz * layer_size) + (stack_top.y * row_size) + tx ));

      // perpendicular bisector of obstacles stack1 and stack2 intersects y-column at (tx, intersection, tz)
      intersection = devicePBAintersection3DY(stack_second.x, stack_top.y, stack_second.z, stack_top.x, last_y, stack_top.z, tx, tz);
    }

    s_stack_second[threadIdx.x] = stack_second;
    s_stack_top[threadIdx.x] = stack_top;
    s_last_y[threadIdx.x] = last_y;
    s_intersection[threadIdx.x] = intersection;
  }

  __syncthreads();

  for (int ty = dims.y - 1 - band_id; ty >= 0; ty -= band_count) { //band_id is 0..(m3-1); band_count is m3
    //all m3 threads read same shared value!; the larger m3, the longer the while loop will need to run, causing lots of divergence
    //optimise: find optimal m3; minimise divergence?
    stack_second = s_stack_second[threadIdx.x];
    stack_top = s_stack_top[threadIdx.x];
    last_y = s_last_y[threadIdx.x];
    intersection = s_intersection[threadIdx.x];

    //performance: the fewer different voronoi regions intersect the part of the y-column processed by this group of m3 threads, the less iterations in this while loop
    while (stack_top.y != PBA_UNINITIALISED_COORD) { // update intersection, stack_top/second until ty > intersection
      if (ty > intersection)
        break; //don't update intersection unless ty <= intersection

      // how to treat ty == intersection? want to save smaller coordinates!
      // what if ty == 0 == intersection? -> then stack_second.y is invalid and no new value will get loaded

      // else: forget stack_top (top of stack); ty has "crossed" the intersector; also update intersection
      //      top=second, second = *(second.backpointer); intersection = ...

      last_y = stack_top.y;
      stack_top = stack_second;

      if (stack_top.y != PBA_UNINITIALISED_COORD) {
        stack_second = (DistanceVoxel)(tex1Dfetch<int>(stack, (tz * layer_size) + (stack_top.y * row_size) + tx ));
        intersection = devicePBAintersection3DY(stack_second.x, stack_top.y, stack_second.z, stack_top.x, last_y, stack_top.z, tx, tz);
      }
    }

    __syncthreads();

    // coalesced global write
    distance_map[(tz * layer_size) + (ty * row_size) + tx] = DistanceVoxel(Vector3ui(stack_top.x, last_y, stack_top.z));

    // thread with max band_id has minmal ty -> most accurate info: update shared values
    if (band_id == band_count - 1) {
      s_stack_second[threadIdx.x] = stack_second; s_stack_top[threadIdx.x] = stack_top; s_last_y[threadIdx.x] = last_y; s_intersection[threadIdx.x] = intersection;
    }

    __syncthreads();
  }
}

/**
 * use shared memory; prevent bank conflicts when warp accesses y-neighbors
 * transpose in-place
 *
 * requires dims.x == dims.y
 *
 * load two shared memory tiles for blockIdx (x,y) and (y,x), switch voxels x/y info and write one tiles info to the others origin
 * special case blockIdx x==y: load tile1, switch x/y and write back only once!
 */
template<typename InputIterator>
__global__
void kernelPBA3DTransposeXY(InputIterator voxels){

  __shared__ uint3 tile1[PBA_TILE_DIM][PBA_TILE_DIM+1]; //+1 padding per row leads to guaranteed offset when same warp accesses y-neighbors
  __shared__ uint3 tile2[PBA_TILE_DIM][PBA_TILE_DIM+1];

  //optimise: could reduce thread count by processing several z-layers in one thread? similar to PBA

  //ensure every tile is transposed just once; all threads within a tile (block) will follow same branch
  //important: if blockIdx x==y, do not load, transpose and store twice!
  if (blockIdx.x > blockIdx.y)
    return;

  const int row_size = gridDim.x * PBA_TILE_DIM;
  const int layer_size = row_size * (gridDim.y * PBA_TILE_DIM);

  // first block coordinates
  // PBA_TILE_DIM == blockDim.x == blockDim.y
  int x1 = blockIdx.x * PBA_TILE_DIM + threadIdx.x;
  int y1 = blockIdx.y * PBA_TILE_DIM + threadIdx.y;
  int z1 = blockIdx.z;

  // load tile1
  //tile[threadIdx.y][threadIdx.x] = idata[y1*width + x1] and transpose x/y
  {
    //TODO: refactor to store DVs?
    DistanceVoxel voxel = (DistanceVoxel)voxels[ (z1 * layer_size) + (y1 * row_size) + x1];
    uint3 o = voxel.getObstacle();
    uint3 xy_transposed = make_uint3(o.y, o.x, o.z); //switch obstacle xy coordinates so they remain valid
    tile1[threadIdx.y][threadIdx.x] = xy_transposed;
  }

  // second (transposed) block coordinates
  int x2 = blockIdx.y * PBA_TILE_DIM + threadIdx.x;  // transpose block x-offset is now blockIdx.y * TILE_DIM instead of blockIdx.x * TILE_DIM
  int y2 = blockIdx.x * PBA_TILE_DIM + threadIdx.y;
  int z2 = z1;

  // load tile2
  {
    if (blockIdx.x != blockIdx.y) {
      DistanceVoxel voxel = (DistanceVoxel)voxels[ (z2 * layer_size) + (y2 * row_size) + x2];
      uint3 obstacle = voxel.getObstacle();
      uint3 xy_transposed = make_uint3(obstacle.y, obstacle.x, obstacle.z); //switch obstacle xy coordinates so they remain valid
      tile2[threadIdx.y][threadIdx.x] = xy_transposed;
    }
  }

  __syncthreads();

  //store tile1
  {
    uint3 tmp = tile1[threadIdx.x][threadIdx.y];
    voxels[ (z2 * layer_size) + (y2 * row_size) + x2] = DistanceVoxel(tmp);
  }

  //store tile2
  {
    if (blockIdx.x != blockIdx.y) {
      uint3 tmp = tile2[threadIdx.x][threadIdx.y];
      voxels[ (z1 * layer_size) + (y1 * row_size) + x1] = DistanceVoxel(tmp);
    }
  }
}

} // end of namespace voxelmap
} // end of namespace gpu_voxels
#endif
