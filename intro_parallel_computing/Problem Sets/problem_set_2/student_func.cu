//==================================================================================
// Homework 2
// Date: Dec 3, 2016
// Student: Trung C. Nguyen
// Image Blurring
//==================================================================================
// In this homework we are blurring an image. To do this, imagine that we have
// a square array of weight values. For each pixel in the image, imagine that we
// overlay this square array of weights on top of the image such that the center
// of the weight array is aligned with the current pixel. To compute a blurred
// pixel value, we multiply each pair of numbers that line up. In other words, we
// multiply each weight with the pixel underneath it. Finally, we add up all of the
// multiplied numbers and assign that value to our output for the current pixel.
// We repeat this process for all the pixels in the image.

// To help get you started, we have included some useful notes here.

//****************************************************************************

// For a color image that has multiple channels, we suggest separating
// the different color channels so that each color is stored contiguously
// instead of being interleaved. This will simplify your code.

// That is instead of RGBARGBARGBARGBA... we suggest transforming to three
// arrays (as in the previous homework we ignore the alpha channel again):
//  1) RRRRRRRR...
//  2) GGGGGGGG...
//  3) BBBBBBBB...
//
// The original layout is known an Array of Structures (AoS) whereas the
// format we are converting to is known as a Structure of Arrays (SoA).

// As a warm-up, we will ask you to write the kernel that performs this
// separation. You should then write the "meat" of the assignment,
// which is the kernel that performs the actual blur. We provide code that
// re-combines your blurred results for each color channel.

//****************************************************************************

// You must fill in the gaussian_blur kernel to perform the blurring of the
// inputChannel, using the array of weights, and put the result in the outputChannel.

// Here is an example of computing a blur, using a weighted average, for a single
// pixel in a small image.
//
// Array of weights:
//
//  0.0  0.2  0.0
//  0.2  0.2  0.2
//  0.0  0.2  0.0
//
// Image (note that we align the array of weights to the center of the box):
//
//    1  2  5  2  0  3
//       -------
//    3 |2  5  1| 6  0       0.0*2 + 0.2*5 + 0.0*1 +
//      |       |
//    4 |3  6  2| 1  4   ->  0.2*3 + 0.2*6 + 0.2*2 +   ->  3.2
//      |       |
//    0 |4  0  3| 4  2       0.0*4 + 0.2*0 + 0.0*3
//       -------
//    9  6  5  0  3  9
//
//         (1)                         (2)                 (3)
//
// A good starting place is to map each thread to a pixel as you have before.
// Then every thread can perform steps 2 and 3 in the diagram above
// completely independently of one another.

// Note that the array of weights is square, so its height is the same as its width.
// We refer to the array of weights as a filter, and we refer to its width with the
// variable filterWidth.
// NOTE:
//    1st version                                   :  1.813952 ms
//    2nd version ( __shared__ filter_local)        :  1.511 ms
//    3rd version ( __shared__ inputChannel_local)  :   NOT DONE
//
//
// ---------------------------------------------------------------
// Mistake that I made:
//    + Size of filter is 9 x 9 instead of 3 x 3
//***************************************************************************************

/****************************************************************************************
 *---------------- DEVICE FUNCTIONS -----------------------------------------------------
 ****************************************************************************************/
#include "utils.h"
#include <stdio.h>

#define  A  blockDim.x
#define  B  blockDim.y

#define top_condition     ( threadIdx.y < halfWidthFilter )
#define bottom_condition  ((threadIdx.y >= B - halfWidthFilter) && (threadIdx.y < B)) || \
                              ((y >= numRows - halfWidthFilter) && (y < numRows)) 

#define left_condition    (threadIdx.x < halfWidthFilter) 
#define right_condition   ((threadIdx.x >= A - halfWidthFilter) && (threadIdx.x < A)) || \
                              ((x >= numCols - halfWidthFilter) && (x < numCols))

__global__
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const       outputChannel,
                   int                        numRows, 
                   int                        numCols,
                   const float* const         filter, 
                   const int                  filterWidth)
{
    // TODO
    int const x           = threadIdx.x + blockDim.x * blockIdx.x;
    int const y           = threadIdx.y + blockDim.y * blockIdx.y;
    int const offset_1D   = x + y * numCols;
    // Any thread working on unbounded pixel of picture will be neglected
    if (x >= numCols  || y >= numRows){
        return;
    }

    //////////////// 2nd Version ////////////////////////
    // Keep filter into __shared__ mem filter_local
    extern __shared__ float shared_filter[];

    int const offset_1D_in_block =  threadIdx.x + threadIdx.y * blockDim.x;

    if( offset_1D_in_block >= 0 && offset_1D_in_block < filterWidth * filterWidth){
      shared_filter[offset_1D_in_block] = filter[offset_1D_in_block];
    }
    __syncthreads();
    /////////////////////////////////////////////////////

    extern __shared__ unsigned char shared_inputChannel[]; // size 24 x 24 x 4 bytes
    int  halfWidthFilter  = (int)(filterWidth/2);
    int shared_inputChannel_width = halfWidthFilter * 2 + blockDim.x;  // working with odd number of filterWidth
    //////////////// 3rd Version ////////////////////////
    // Keep inputChannel into __shared__ memory
    // --- Copy pixel overlay block --
    int shared_inputChannel_x = threadIdx.x + halfWidthFilter;
    int shared_inputChannel_y = threadIdx.y + halfWidthFilter;

    shared_inputChannel[ shared_inputChannel_y * shared_inputChannel_width + shared_inputChannel_x ] = inputChannel[offset_1D];

    /*  
    int safe_image_y;
    int safe_image_x;
    // --- Copy pixels around block
        //-- Top 
    if( top_condition ){
        safe_image_x          = x;
        safe_image_y          = max(y - halfWidthFilter, 0);

        shared_inputChannel_x = threadIdx.x + halfWidthFilter;
        shared_inputChannel_y = threadIdx.y ;

        shared_inputChannel[shared_inputChannel_y * shared_inputChannel_width + shared_inputChannel_x] 
            = inputChannel[safe_image_y * numCols + safe_image_x];
    }

        //-- Bottom
    if( bottom_condition ){
        shared_inputChannel_x = threadIdx.x + halfWidthFilter;
        shared_inputChannel_y = threadIdx.y + halfWidthFilter;

        safe_image_x          = x;
        safe_image_y          = min(y + halfWidthFilter, numRows - 1);

        shared_inputChannel[shared_inputChannel_y * shared_inputChannel_width + shared_inputChannel_x] 
            = inputChannel[safe_image_y * numCols + safe_image_x];
    }
        //-- Left
    if( left_condition ){
        shared_inputChannel_x = threadIdx.x;
        shared_inputChannel_y = threadIdx.y + halfWidthFilter;

        safe_image_x          = max(0, x - halfWidthFilter);
        safe_image_y          = y;

        shared_inputChannel[shared_inputChannel_y * shared_inputChannel_width + shared_inputChannel_x] 
            = inputChannel[safe_image_y * numCols + safe_image_x];
    }
        //-- Right
    if( right_condition ){
        shared_inputChannel_x = threadIdx.x + halfWidthFilter;
        shared_inputChannel_y = threadIdx.y + halfWidthFilter;

        safe_image_x          = min(numCols - 1, x + halfWidthFilter);
        safe_image_y          = y;

        shared_inputChannel[shared_inputChannel_y * shared_inputChannel_width + shared_inputChannel_x] 
            = inputChannel[safe_image_y * numCols + safe_image_x];
    }
    //---------------------------------------------------------------//
        //-- Left + Top corner
    if( top_condition && left_condition ){
        shared_inputChannel_x = threadIdx.x;
        shared_inputChannel_y = threadIdx.y;

        safe_image_x          = max(0, x - halfWidthFilter);
        safe_image_y          = max(y - halfWidthFilter, 0);

        shared_inputChannel[shared_inputChannel_y * shared_inputChannel_width + shared_inputChannel_x] 
            = inputChannel[safe_image_y * numCols + safe_image_x];
    }

        //-- Right + Top corner
    if( top_condition && right_condition ){
        shared_inputChannel_x = threadIdx.x + halfWidthFilter;
        shared_inputChannel_y = threadIdx.y;

        safe_image_x          = min(numCols - 1, x + halfWidthFilter);
        safe_image_y          = max(y - halfWidthFilter, 0);

        shared_inputChannel[shared_inputChannel_y * shared_inputChannel_width + shared_inputChannel_x] 
            = inputChannel[safe_image_y * numCols + safe_image_x];
    }

        //-- Left + Bottom corner
    if( bottom_condition && left_condition ){
        shared_inputChannel_x = threadIdx.x;
        shared_inputChannel_y = threadIdx.y + halfWidthFilter;

        safe_image_x          = max(0, x - halfWidthFilter);
        safe_image_y          = min(y + halfWidthFilter, numRows - 1);

        shared_inputChannel[shared_inputChannel_y * shared_inputChannel_width + shared_inputChannel_x] 
            = inputChannel[safe_image_y * numCols + safe_image_x];
    }

        //-- Right + Bottom corner
    if( bottom_condition && right_condition ){
        shared_inputChannel_x = threadIdx.x + halfWidthFilter;
        shared_inputChannel_y = threadIdx.y + halfWidthFilter;

        safe_image_x          = min(numCols - 1, x + halfWidthFilter);
        safe_image_y          = min(y + halfWidthFilter, numRows - 1);

        shared_inputChannel[shared_inputChannel_y * shared_inputChannel_width + shared_inputChannel_x] 
            = inputChannel[safe_image_y * numCols + safe_image_x];
    }
    __syncthreads();
    */
    /////////////////////////////////////////////////////
    float result = 0.f;
    /*
    //--- Version 1 + 2
    for (int filter_r = -filterWidth/2; filter_r <= filterWidth/2; ++filter_r) {
        for (int filter_c = -filterWidth/2; filter_c <= filterWidth/2; ++filter_c) {
    
            int safe_image_y = min(max(y + filter_r, 0), numRows - 1);
            int safe_image_x = min(max(x + filter_c, 0), numCols - 1);

            float image_value = (float)(inputChannel[safe_image_y * numCols + safe_image_x]);
            float filter_value = shared_filter[(filter_r + filterWidth/2) * filterWidth + filter_c + filterWidth/2];

            result += image_value * filter_value;
         }
    }
    */

    ///* 
    //--Version 3
    for (int filter_r = -halfWidthFilter; filter_r <= halfWidthFilter; ++filter_r) {
        for (int filter_c = -halfWidthFilter; filter_c <= halfWidthFilter; ++filter_c) {
    
            int image_y = y + filter_r + halfWidthFilter;
            int image_x = x + filter_c + halfWidthFilter;

            float image_value = (float)(shared_inputChannel[image_y * shared_inputChannel_width + image_x]);
            float filter_value = shared_filter[(filter_r + halfWidthFilter) * filterWidth + filter_c + halfWidthFilter];

            result += image_value * filter_value;
         }
    }
    //*/
    outputChannel[offset_1D] = (unsigned char)result;
}
/****************************************************************************************
 * This kernel takes in an image represented as a uchar4 and splits
 * it into three images consisting of only one color channel each
 *
 ****************************************************************************************/
__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int                 numRows,
                      int                 numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{

  // Coding for arbitrary number of blocks
  int const x           = threadIdx.x + blockIdx.x * blockDim.x;
  int const y           = threadIdx.y + blockIdx.y * blockDim.y;

  int const offset_1D   = x + y * numCols;

  // NOTE: Be careful not to try to access memory that is outside the bounds of
  // the image.
  if( x >= numCols || y >= numRows ){
     return;
  }


  redChannel[offset_1D]     = inputImageRGBA[offset_1D].x;
  greenChannel[offset_1D]   = inputImageRGBA[offset_1D].y;
  blueChannel[offset_1D]    = inputImageRGBA[offset_1D].z;
}

/****************************************************************************************
 * This kernel takes in three color channels and recombines them
 * into one image.  The alpha channel is set to 255 to represent
 * that this image has no transparency.
 ****************************************************************************************/
__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const              outputImageRGBA,
                       int                        numRows,
                       int                        numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}


/****************************************************************************************
 *  HOST FUNCTION
 *  Allocate GPU memory for filter
 *  Copy weights of filter from Host memory to Device memory
 *  @param
 *      numRowsImage    : Constant value
 *      numColsImage    : Constant value
 *      h_filter        : Constant pointer to a const float  
 *                        (the same as "float const * const h_filter)
 *      filterWidth     : dimension of filter
 *  @return
 *      None
 *  @note
 *       None
 ****************************************************************************************/
// variables in GPU global memory
unsigned char *d_red, *d_green, *d_blue;
float *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{

  //allocate memory for the three different channels
  //original
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

  //TODO:
  //Allocate memory for the filter on the GPU
  //IMPORTANT: Notice that we pass a pointer to a pointer to cudaMalloc
  checkCudaErrors( cudaMalloc( &d_filter, sizeof(float) * filterWidth * filterWidth) );

  //TODO:
  //Copy the filter on the host (h_filter) to the memory you just allocated
  //on the GPU.  cudaMemcpy(dst, src, numBytes, cudaMemcpyHostToDevice);
  //Remember to use checkCudaErrors!
  checkCudaErrors( cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice) );
}

/****************************************************************************************
 *  HOST FUNCTION
 *  Allocate GPU memory for filter
 *  Copy weights of filter from Host memory to Device memory
 *  @param
 *      numRowsImage    : Constant value
 *      numColsImage    : Constant value
 *      h_filter        : Constant pointer to a const float  
 *                        (the same as "float const * const h_filter)
 *      filterWidth     : dimension of filter
 *  @return
 *      None
 *  @note
 *       None
 ****************************************************************************************/
void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred, 
                        unsigned char *d_greenBlurred, 
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
  //TODO: Set reasonable block size (i.e., number of threads per block)
  const dim3 blockSize(16, 16, 1);
  //TODO:
  //Compute correct grid size (i.e., number of blocks per kernel launch)
  //from the image size and and block size.
  const dim3 gridSize((numCols + blockSize.x - 1)/blockSize.x, (numRows + blockSize.y - 1)/blockSize.y, 1 );

  const int shared_memSize = ( filterWidth * filterWidth  + ( blockSize.x + (int)(filterWidth - 1) ) * ( blockSize.y + (int)(filterWidth - 1) ) ) * sizeof(float);
  //TODO: Launch a kernel for separating the RGBA image into different color channels
  separateChannels<<< gridSize, blockSize  >>>(  d_inputImageRGBA,
                                                numRows,
                                                numCols,
                                                d_red,
                                                d_green,
                                                d_blue
                                              );
  // Call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //TODO: Call your convolution kernel here 3 times, once for each color channel.
  gaussian_blur<<< gridSize, blockSize, shared_memSize >>>(   d_red,
                                                              d_redBlurred,
                                                              numRows, numCols,
                                                              d_filter,
                                                              filterWidth);
  gaussian_blur<<< gridSize, blockSize, shared_memSize >>>(   d_green,
                                                              d_greenBlurred,
                                                              numRows, numCols,
                                                              d_filter,
                                                              filterWidth);
  gaussian_blur<<< gridSize, blockSize, shared_memSize >>>(   d_blue,
                                                              d_blueBlurred,
                                                              numRows, numCols,
                                                              d_filter,
                                                              filterWidth);
  // Again, call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // Now we recombine your results. We take care of launching this kernel for you.
  //
  // NOTE: This kernel launch depends on the gridSize and blockSize variables,
  // which you must set yourself.
  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred,
                                             d_greenBlurred,
                                             d_blueBlurred,
                                             d_outputImageRGBA,
                                             numRows,
                                             numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}

/****************************************************************************************
 *  HOST FUNCTION
 *  Free all the memory that we allocated
 *  @param
 *      None
 *  @return
 *      None
 *  @note
 *       None
 ****************************************************************************************/
void cleanup() {
  checkCudaErrors( cudaFree(d_red) );
  checkCudaErrors( cudaFree(d_green) );
  checkCudaErrors( cudaFree(d_blue) );
  checkCudaErrors( cudaFree(d_filter) );
}

