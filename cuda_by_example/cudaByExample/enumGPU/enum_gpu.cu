/*
 * Trung C. Nguyen - TU Kaiserslautern
 * Main code getting from "Cuda by Example"
 * Enumurate properties of GPU
 * Important APIs:
 *  - cudaGetDeviceCount()   : Get number of GPU on a system on CUDA architecture
 *  - cudaGetDeviceProperties( &pro_struct, #device) : Get property of GPU number "#device"
 * Property structures:
 *  - cudaDeviceProp         : Structure keeping properties of a GPU
 *      char name[256]              : An ASCII string identifying the device (e.g., "GeForce GTX 280")
        size_t totalGlobalMem       : The amount of global memory on the device in bytes
        size_t sharedMemPerBlock    : The maximum amount of shared memory a single block may use in bytes
        int regsPerBlock            : The number of 32-bit registers available per block
        int warpSize                : The number of threads in a warp
        size_t memPitch             : The maximum pitch allowed for memory copies in bytes
        int maxThreadsPerBlock      : The maximum number of threads that a block may contain
        int maxThreadsDim[3]        : The maximum number of threads allowed along each dimension of a block
        int maxGridSize[3]          : The number of blocks allowed along each dimension of a grid
        size_t totalConstMem        : The amount of available constant memory
        int major                   : The major revision of the device’s compute capability
        int minor                   : The minor revision of the device’s compute capability
        size_t textureAlignment     : The device’s requirement for texture alignment
        int deviceOverlap           : A boolean value representing whether the device can simultaneously perform a cudaMemcpy() and kernel execution
        int multiProcessorCount     : The number of multiprocessors on the device
        int kernelExecTimeoutEnabled: A boolean value representing whether there is a runtime limit for kernels executed on this device
        int integrated              : A boolean value representing whether the device is an integrated GPU (i.e., part of the chipset and not a discrete GPU)
        int canMapHostMemory        : A boolean value representing whether the device can map host memory into the CUDA device address space
        int computeMode             : A value representing the device’s computing mode: default, exclusive, or prohibited
        int maxTexture1D            : The maximum size supported for 1D textures
        int maxTexture2D[2]         : The maximum dimensions supported for 2D textures
        int maxTexture3D[3]         : The maximum dimensions supported for 3D textures
        int maxTexture2DArray[3]    : The maximum dimensions supported for 2D texture arrays
        int concurrentKernels       : A boolean value representing whether the device supports executing multiple kernels within the same context simultaneously
 */


#include "../common/book.h"

int main( void ) {
    /********************************************************/
    // Find: How many GPUs in the system on CUDA architecture
    /********************************************************/
    int count;
    HANDLE_ERROR( cudaGetDeviceCount( &count ) );

    /********************************************************/
    // Get properties of each GPU
    /********************************************************/
    // Structure keeps properties of a GPU
    cudaDeviceProp  prop;

    for (int i=0; i< count; i++) {
        HANDLE_ERROR( cudaGetDeviceProperties( &prop, i ) );
        printf( "==================================================\n");
        printf( "   --- General Information for device %d ---\n", i+1 );
        printf( "Name (GPU):  %s\n", prop.name );
        printf( "Compute capability:  %d.%d\n", prop.major, prop.minor );
        printf( "Clock rate:  %d\n", prop.clockRate );
        printf( "Device copy overlap (overlap cudaMemcpy() with kernel execution) :  " );
        if (prop.deviceOverlap)
            printf( "Enabled\n" );
        else
            printf( "Disabled\n");
        printf( "Kernel execution timeout :  " );
        if (prop.kernelExecTimeoutEnabled)
            printf( "Enabled\n" );
        else
            printf( "Disabled\n" );

        printf( "   --- Memory Information for device %d (in bytes)---\n", i+1 );
        printf( "Total global mem:  %ld\n", prop.totalGlobalMem );
        printf( "Total constant Mem:  %ld\n", prop.totalConstMem );
        printf( "Max mem pitch:  %ld\n", prop.memPitch );
        printf( "Texture Alignment:  %ld\n", prop.textureAlignment );

        printf( "   --- MP Information for device %d ---\n", i+1 );
        printf( "Multiprocessor count:  %d\n",
                    prop.multiProcessorCount );
        printf( "Shared mem per block:  %ld\n", prop.sharedMemPerBlock );
        printf( "Registers per block:  %d\n", prop.regsPerBlock );
        printf( "Threads in warp:  %d\n", prop.warpSize );
        printf( "Max threads per block:  %d\n",
                    prop.maxThreadsPerBlock );
        printf( "Max thread dimensions:  (%d, %d, %d)\n",
                    prop.maxThreadsDim[0], prop.maxThreadsDim[1],
                    prop.maxThreadsDim[2] );
        printf( "Max grid dimensions:  (%d, %d, %d)\n",
                    prop.maxGridSize[0], prop.maxGridSize[1],
                    prop.maxGridSize[2] );
        printf( "\n" );
    }
}
