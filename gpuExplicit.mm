// gpuExplicit.mm
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <chrono>


struct HeatParams {
    uint32_t numXSteps;
    uint32_t numYSteps;
    float    alpha;
    float    dtOverDxSq;
    float    dtOverDySq;
};

double timeLowerBoundary(int x, int y, float dx, float dy, int xMin, int yMin)
{
    return sin(M_PI * x * dx + xMin) * sin(M_PI * y * dy + yMin);
}

// closed form solution
double calculateExactSol(int i, int j, float T, float dx, float dy)
{
    // return (np.e ** (-2 * (np.pi**2) * t)) * np.sin(np.pi*x * dx) * np.sin(np.pi*y * dy)
    return std::exp(-2 * (M_PI * M_PI) * T) * std::sin(M_PI * i * dx) * std::sin(M_PI * j * dy);
}
// L^2 error
double computeError(const std::vector<float> &uNumeric, HeatParams& params, float T, float dx, float dy)
{
    double ans = 0;
    for (int i = 0; i < params.numXSteps; i++)
    {
        for (int j = 0; j < params.numYSteps; j++)
        {
            float diff = calculateExactSol(i, j, T, dx, dy) - uNumeric[i * params.numXSteps + j];
            ans += (diff * diff) * dx * dy;
        }
    }
    return std::sqrt(ans);
}

int main() {
    @autoreleasepool {

        const float T{0.1}, dx{1.0 / 512}, dy{1.0 / 512}, dt{8.0e-7}, alpha{1};
        const int xMin{0}, xMax{1}, yMin{0}, yMax{1};
        const uint32_t numTimeSteps{static_cast<uint32_t>(T / dt)}, numXSteps{static_cast<uint32_t>((xMax - xMin) / dx) + 1}, numYSteps{static_cast<uint32_t>((yMax - yMin) / dy) + 1};

        const float dtOverDxSq = dt / (dx * dx);
        const float dtOverDySq = dt / (dy * dy);


        const int numPoints = numXSteps * numYSteps;

        std::vector<float> uXyPrev(numPoints, 0.0f);
        std::vector<float> uXyCurr(numPoints, 0.0f);

        // Initial condition: u(x,y,0) = sin(pi x) sin(pi y)
        for (uint32_t i = 1; i < numXSteps - 1; i++) {
            for (uint32_t j = 1; j < numYSteps - 1; j++) {
                size_t idx = static_cast<size_t>(i) * numYSteps + j;
                uXyPrev[idx] = timeLowerBoundary(i, j, dx, dy, xMin, yMin);
            }
        }


        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Failed to get default Metal device.\n";
            return 1;
        }

        NSError *error = nil;
        NSString *libPath = [NSString stringWithUTF8String:"heat_kernels.metallib"];

        id<MTLLibrary> library = [device newLibraryWithFile:libPath error:&error];
        if (!library) {
            std::cerr << "Failed to load heat_kernels.metallib: "
                      << (error ? [[error localizedDescription] UTF8String] : "unknown error")
                      << "\n";
            return 1;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"heat_step"];
        if (!function) {
            std::cerr << "Failed to find kernel function 'heat_step' in metallib.\n";
            return 1;
        }

        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            std::cerr << "Failed to create compute pipeline: "
                      << (error ? [[error localizedDescription] UTF8String] : "unknown error")
                      << "\n";
            return 1;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            std::cerr << "Failed to create command queue.\n";
            return 1;
        }


        const MTLResourceOptions bufOpts = MTLResourceStorageModeManaged;

        id<MTLBuffer> uPrevBuf =
            [device newBufferWithLength:numPoints * sizeof(float)
                                 options:bufOpts];
        id<MTLBuffer> uNextBuf =
            [device newBufferWithLength:numPoints * sizeof(float)
                                 options:bufOpts];

        if (!uPrevBuf || !uNextBuf) {
            std::cerr << "Failed to allocate Metal buffers.\n";
            return 1;
        }

        // Copy initial data to GPU
        std::memcpy([uPrevBuf contents], uXyPrev.data(), numPoints * sizeof(float));
        [uPrevBuf didModifyRange:NSMakeRange(0, numPoints * sizeof(float))];

        HeatParams params = { numXSteps, numYSteps, alpha, dtOverDxSq, dtOverDySq };
        id<MTLBuffer> paramsBuf =
            [device newBufferWithBytes:&params
                                length:sizeof(HeatParams)
                               options:bufOpts];
        if (!paramsBuf) {
            std::cerr << "Failed to allocate params buffer.\n";
            return 1;
        }

        MTLSize gridSize = MTLSizeMake(numYSteps, numXSteps, 1); // (width=Ny, height=Nx)
        // Choose a reasonable threadgroup size (e.g., 16x16)
        const NSUInteger tgSizeX = 16;
        const NSUInteger tgSizeY = 16;
        MTLSize threadgroupSize =
            MTLSizeMake(tgSizeX, tgSizeY, 1);

        // Compute number of threadgroups
        MTLSize numThreadgroups = MTLSizeMake(
            (gridSize.width  + tgSizeX - 1) / tgSizeX,
            (gridSize.height + tgSizeY - 1) / tgSizeY,
            1
        );


        auto t0 = std::chrono::steady_clock::now();
        for (int step = 1; step < numTimeSteps; step++) {
            id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

            [enc setComputePipelineState:pipeline];
            [enc setBuffer:uPrevBuf offset:0 atIndex:0];
            [enc setBuffer:uNextBuf offset:0 atIndex:1];
            [enc setBuffer:paramsBuf offset:0 atIndex:2];

            [enc dispatchThreadgroups:numThreadgroups
                 threadsPerThreadgroup:threadgroupSize];
            [enc endEncoding];

            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];

            // Swap buffers for next step
            std::swap(uPrevBuf, uNextBuf);
        }
        auto t1 = std::chrono::steady_clock::now();

  
        std::memcpy(uXyPrev.data(), [uPrevBuf contents], numPoints * sizeof(float));
        std::cout << "GPU time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << std::endl;
        std::cout << "GPU L^2 error: " << computeError(uXyPrev, params, T, dx, dy) << std::endl;


        uint32_t midI = numXSteps / 2;
        uint32_t midJ = numYSteps / 2;
        size_t midIdx = static_cast<size_t>(midI) * numYSteps + midJ;


        return 0;
    }
}