#!/bin/bash

if [ $# -ne 2 ]
  then
    echo "Usage: voxelize.sh path_to_mesh_file scaling_factor"
    exit 1
fi

PATH=$PATH:.

MESH_FILE="$1"
SCALING="$2"

echo "==================================================="
echo "====== Step 1: Executing BinVox prerun... ========="
echo "==================================================="

PREVOX_OUTPUT=$(binvox -d 20 "$MESH_FILE")

echo "         --------- Pre-Run Output -----------"
echo "$PREVOX_OUTPUT"
echo "         -------- END Pre-Run Output --------"
echo ""


read LENGTH <<< $(awk '/longest length:/ { print $3 }' <<< "$PREVOX_OUTPUT")

CLEANED=$(sed -e 's/,//g' -e 's/\[//g' <<< "$PREVOX_OUTPUT")
read MIN_X MIN_Y MIN_Z MAX_X MAX_Y MAX_Z <<< $(awk '/Mesh::normalize bounding box:/ {print $4" "$5" "$6" "$9" "$10" "$11 }' <<< "$CLEANED")
read DEL_FILENAME <<< $(sed -e 's/^.*(//;s/)$//' <<< $(awk '/VoxelFile::write_file\(/ {print $1 }' <<< "$PREVOX_OUTPUT"))

rm $DEL_FILENAME

#This is a ceil function:
NUM_VOXELS=$(bc <<< "sqrt( ( ($LENGTH + $SCALING) / $SCALING )^2 )")

echo "==================================================="
echo "====== Step 2: Executing BinVox final run... ======"
echo "==================================================="
echo "Params are:"
echo "Bounding Box of input mesh: Min: [ ${MIN_X}, ${MIN_Y}, ${MIN_Z} ] Max: [ ${MAX_X}, ${MAX_Y}, ${MAX_Z} ]"
echo "Bounding Box maximum side length: ${LENGTH} ==> Voxelcube sidelength in Voxels: ${NUM_VOXELS}"
echo ""

echo "         --------- Final-Run Output -----------"
#Add the -e option to generate hollow models.
echo "binvox -d $NUM_VOXELS -bb ${MIN_X} ${MIN_Y} ${MIN_Z} ${MAX_X} ${MAX_Y} ${MAX_Z} $MESH_FILE"
binvox -d $NUM_VOXELS -bb ${MIN_X} ${MIN_Y} ${MIN_Z} ${MAX_X} ${MAX_Y} ${MAX_Z} $MESH_FILE
echo "         -------- END Final-Run Output --------"
