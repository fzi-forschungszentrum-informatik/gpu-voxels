#include "gvl_linkage_test_lib.h"


gvlLinkageTest::gvlLinkageTest()
{
    gvl = gpu_voxels::GpuVoxels::getInstance();
    gvl->initialize(150, 150, 150, 0.01);

    // We add maps with objects, to collide them
    gvl->addMap(gpu_voxels::MT_PROBAB_VOXELMAP,"myFirstMap");
    gvl->addMap(gpu_voxels::MT_PROBAB_VOXELMAP,"mySecondMap");

}

gvlLinkageTest::~gvlLinkageTest()
{
    gvl.reset(); // Not even required, as we use smart pointers.
}

int gvlLinkageTest::doFancyStuff()
{
    gpu_voxels::Vector3f center = gpu_voxels::Vector3f(0.5 , 0.5, 0.5);
    gpu_voxels::Vector3f box_size = gpu_voxels::Vector3f(0.4 , 0.4, 0.4);
    gpu_voxels::Vector3f overlap = gpu_voxels::Vector3f(0.1 , 0.1, 0.1);

    gpu_voxels::Vector3f c1 = center - overlap;
    gpu_voxels::Vector3f c2 = c1 + box_size;

    gpu_voxels::Vector3f c3 = center + overlap;
    gpu_voxels::Vector3f c4 = c3 - box_size;

    gvl->insertBoxIntoMap(c1, c2, "myFirstMap", gpu_voxels::eBVM_OCCUPIED, 1);
    gvl->insertBoxIntoMap(c4, c3, "mySecondMap", gpu_voxels::eBVM_OCCUPIED, 1);

    size_t num_colls_pc = gvl->getMap("myFirstMap")->as<gpu_voxels::voxelmap::ProbVoxelMap>()->collideWith(gvl->getMap("mySecondMap")->as<gpu_voxels::voxelmap::ProbVoxelMap>());
    LOGGING_INFO(gpu_voxels::Gpu_voxels, num_colls_pc << " collisions detected" << gpu_voxels::endl );

    return 0;
}
