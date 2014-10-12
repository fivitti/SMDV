/*  Kernel CSR
 *  @author: Krzysztof Sopyla
 *  Source: KMLib [https://github.com/ksirg/KMLib]
 */
texture<float, 1, cudaReadModeElementType> mainVecTexRef;

extern "C" __global__ void rbfCsrFormatKernel(const float * vals,
                                       const int * idx, 
                                       const int * vecPointers, 
                                       float * results,
                                       const int num_rows)
{
    __shared__ float sdata[{{ BLOCK_SIZE }} + 16];                    // padded to avoid reduction ifs
    __shared__ int ptrs[{{ BLOCK_SIZE }}/{{ WARP_SIZE }}][2];
    
    const int thread_id   = {{ BLOCK_SIZE }} * blockIdx.x + threadIdx.x;  // global thread index
    const int thread_lane = threadIdx.x & ({{ WARP_SIZE }}-1);            // thread index within the warp
    const int warp_id     = thread_id   / {{ WARP_SIZE }};                // global warp index
    const int warp_lane   = threadIdx.x / {{ WARP_SIZE }};                // warp index within the CTA
    const int num_warps   = ({{ BLOCK_SIZE }} / {{ WARP_SIZE }}) * gridDim.x;   // total number of active warps

    for(int row = warp_id; row < num_rows; row += num_warps){
        // use two threads to fetch vecPointers[row] and vecPointers[row+1]
        // this is considerably faster than the straightforward version
        if(thread_lane < 2)
            ptrs[warp_lane][thread_lane] = vecPointers[row + thread_lane];
        const int row_start = ptrs[warp_lane][0];            //same as: row_start = vecPointers[row];
        const int row_end   = ptrs[warp_lane][1];            //same as: row_end   = vecPointers[row+1];

        // compute local sum
        float sum = 0;
        for(int jj = row_start + thread_lane; jj < row_end; jj += {{ WARP_SIZE }})
        {
            sum += vals[jj] * tex1Dfetch(mainVecTexRef,idx[jj]);
            //__syncthreads();
        }

        volatile float* smem = sdata;
        smem[threadIdx.x] = sum; __syncthreads(); 
        smem[threadIdx.x] = sum = sum + smem[threadIdx.x + 16]; //__syncthreads(); 
        smem[threadIdx.x] = sum = sum + smem[threadIdx.x +  8]; //__syncthreads();
        smem[threadIdx.x] = sum = sum + smem[threadIdx.x +  4]; //__syncthreads();
        smem[threadIdx.x] = sum = sum + smem[threadIdx.x +  2]; //__syncthreads();
        smem[threadIdx.x] = sum = sum + smem[threadIdx.x +  1]; //__syncthreads();

        // first thread writes warp result
        if (thread_lane == 0){
            results[row]=smem[threadIdx.x];
        }
    }
}