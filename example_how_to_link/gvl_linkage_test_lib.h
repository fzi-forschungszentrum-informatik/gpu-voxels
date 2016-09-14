#ifndef GVL_LINKAGE_TEST_LIB_H_INCLUDED
#define GVL_LINKAGE_TEST_LIB_H_INCLUDED

#include <gpu_voxels/GpuVoxels.h>

class gvlLinkageTest
{
public:
    gvlLinkageTest();
    ~gvlLinkageTest();

    int doFancyStuff();
private:
    gpu_voxels::GpuVoxelsSharedPtr gvl;
};



#endif
