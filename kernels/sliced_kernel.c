/*  Kernel SLICED
 *  @author: Krzysztof Sopyla
 *  Source: KMLib [https://github.com/ksirg/KMLib]
 */
texture<float, 1, cudaReadModeElementType> mainVecTexRef;

extern "C" __global__ void SlicedEllpackFormatKernel(
    const float* vecVals,
    const int* vecCols,
    const int* vecLengths, 
    const int* sliceStart, 
    float* result,
    const int nrRows, 
    const int align)
    {
         __shared__  float sh_cache[{{ sh_cache_size }}];
         int tx = threadIdx.x;
         int txm = tx %  {{ threadPerRow }};
        int thIdx = (blockIdx.x*blockDim.x+tx);
        int row = thIdx/{{ threadPerRow }};  //thIdx>> 2;

        if (row < nrRows){
            float sub = 0.0;
            int maxRow = (int)ceil(vecLengths[row]/(float){{ threadPerRow }});
            int col=-1;
            float value =0.0;
            int idx=0;

            for(int i=0; i < maxRow; i++){
                idx = i*align+sliceStart[blockIdx.x]+tx;
                col     = vecCols[idx];
                value = vecVals[idx];
                sub += value * tex1Dfetch(mainVecTexRef, col);
            }
   
               sh_cache[tx] = sub;
               __syncthreads();
            volatile float *shMem = sh_cache;
   
               for(int s={{ threadPerRow }}/2; s>0; s>>=1) //s/=2
            {
                if(txm < s){
                    shMem[tx] += shMem[tx+s];
                }
            }

            if(txm == 0 ){
                result[row]=sh_cache[tx];
            }
        }//if row<nrRows 
}//end func