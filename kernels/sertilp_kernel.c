/*  Kernel SERTILP
 *  @author: Krzysztof Sopyla, Slawomir Figiel
 *  Source: KMLib [https://github.com/ksirg/KMLib]
 */
texture<float, 1, cudaReadModeElementType> mainVecTexRef;

extern "C" __global__ void SpMV_Sertilp(const float *vecVals,
                                        const int *vecCols,
                                        const int *vecLengths, 
                                        const int * sliceStart, 
                                        float *result,
                                        const int nrRows, 
                                        const int align)
{

    __shared__  float shDot[{{ shDot_size }}];	
    __shared__ int shSliceStart;

    if(threadIdx.x==0)
    {
        shSliceStart=sliceStart[blockIdx.x];
    }
    shDot[threadIdx.x]=0.0f;
    __syncthreads();

    int idxT = threadIdx.x % {{ threadPerRow }}; //thread number in Thread group
    int idxR = threadIdx.x/{{ threadPerRow }}; //row index mapped into block region

    //map group of thread to row, in this case 4 threads are mapped to one row
    int row = (blockIdx.x*blockDim.x+threadIdx.x)/{{ threadPerRow }}; 
    unsigned int j=0;
    if (row < nrRows){
        int maxRow = vecLengths[row];

        float val[{{ prefetch }}];
        int col[{{ prefetch }}];
        float dot[{{ prefetch }}]={0, 0};
   
        unsigned int arIdx=0;
        for(int i=0; i < maxRow; i++){

            #pragma unroll
            for( j=0; j<{{ prefetch }};j++)	{
                arIdx = (i*{{ prefetch }}+j )*align+shSliceStart+threadIdx.x;
                col[j] = vecCols[arIdx];
                val[j] = vecVals[arIdx];
            }

            #pragma unroll
            for( j=0; j<{{ prefetch }};j++){
                dot[j]+=val[j]*tex1Dfetch(mainVecTexRef,col[j]); 
            }
        }

        #pragma unroll
        for( j=1; j<{{ prefetch }};j++){
            dot[0]+=dot[j];	
        }

        shDot[idxT*{{ sliceSize }}+idxR]=dot[0];
        __syncthreads();		
    }

    volatile float *shDotv = shDot;
    //reduction to some level
    for( j=blockDim.x/2; j>={{ sliceSize }}; j>>=1)
    {
        if(threadIdx.x<j){
            shDotv[threadIdx.x]+=shDotv[threadIdx.x+j];
        }
        __syncthreads();
    }

    if(threadIdx.x<{{ sliceSize }}){			
        unsigned int row2=blockIdx.x* {{ sliceSize }}+threadIdx.x;
        if(row2<nrRows){
            result[row2]= shDotv[threadIdx.x];
        }
    }
}//end func