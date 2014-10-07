texture<float,1,cudaReadModeElementType> mainVecTexRef;

__device__ float SpMV_Ellpack(const float * vals,
                                       const int * colIdx, 
                                       const int * rowLength,
                                       const int row,
                                       const int numRows)
{
    const int num_rows =numRows;
    int maxEl = rowLength[row];
    float dot=0;
        
    int col=-1;
    float val=0;
    int i=0;
    for(i=0; i<maxEl;i++)
    {
        col=colIdx[num_rows*i+row];
        val= vals[num_rows*i+row];
        //dot+=val*tex1Dfetch(mainVecTexRef,col);
        dot+=val*tex1Dfetch(mainVecTexRef,col);
    }

    return dot;
}

extern "C" __global__ void EllpackFormatKernel(const float * vals,
                                       const int * colIdx, 
                                       const int * rowLength, 
                                       float * results,
                                       const int numRows)
{
    
    const int row   = blockDim.x * blockIdx.x + threadIdx.x;  // global thread index
    //const int num_rows =numRows;
    if(row<numRows)
    {
        float dot = SpMV_Ellpack(vals,colIdx,rowLength,row,numRows);
        results[row]=dot;
    }	
}

__global__ void copy_texture_kernel(float * data) {
   int ty=threadIdx.x;
   data[ty] =tex1Dfetch(mainVecTexRef, ty);
}