#include <metal_stdlib>
using namespace metal;

struct HeatParams {
    uint32_t numXSteps;
    uint32_t numYSteps;
    float    alpha;
    float    dtOverDxSq;
    float    dtOverDySq;
};

kernel void heat_step(
    device const float* uXyPrev     [[buffer(0)]],
    device float*       uXyCurr     [[buffer(1)]],
    constant HeatParams& params   [[buffer(2)]],
    uint2 gid                    [[thread_position_in_grid]]
) {
    uint i = gid.y; // row    (0 .. numXSteps-1)
    uint j = gid.x; // column (0 .. numYSteps-1)

    if (i == 0 || j == 0 || i >= params.numXSteps - 1 || j >= params.numYSteps - 1) {
        return;
    }

    uint idx = i * params.numYSteps + j;
    uXyCurr[idx] = (params.alpha * params.dtOverDxSq) * (uXyPrev[(i+1) * params.numXSteps + j] + uXyPrev[(i-1) * params.numXSteps + j]) +
                                uXyPrev[idx] * (1 - 2 * params.alpha * params.dtOverDxSq - 2 * params.alpha * params.dtOverDySq) + 
                                params.alpha * params.dtOverDySq * (uXyPrev[i * params.numXSteps + j + 1] + uXyPrev[i * params.numXSteps + j-1]);
}


