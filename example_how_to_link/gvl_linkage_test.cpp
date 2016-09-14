#include <iostream>
using namespace std;
#include <gpu_voxels/logging/logging_gpu_voxels.h>

#include "gvl_linkage_test_lib.h"
#include <stdlib.h>

int main(int argc, char **argv)
{
    // Always initialize our logging framework as a first step, with the CLI arguments:
    icl_core::logging::initialize(argc, argv);

    std::cout << "Creating my functional object" << std::endl;
    gvlLinkageTest my_class;

    return my_class.doFancyStuff();
}
