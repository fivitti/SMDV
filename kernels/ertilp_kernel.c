/*  Kernel ERTILP
 *  @author: Krzysztof Sopyla, Slawomir Figiel
 *  Source: KMLib [https://github.com/ksirg/KMLib]
 */
texture<float,1,cudaReadModeElementType> mainVecTexRef;

extern "C" __global__ void SpMV_Ertilp(const float * vals,
                                       const int * colIdx, 
                                       const int * rowLength, 
                                       float * results,
                                       const int num_rows)
{
    __shared__ int shRows;
    __shared__ float shDot[{{ BLOCK_SIZE }}];
    shDot[threadIdx.x]=0.0;	

    if(threadIdx.x==0)
    {
        shRows = num_rows;
    }
    __syncthreads();

    int row  = (blockDim.x * blockIdx.x + threadIdx.x)/{{ THREADS_ROW }};

    const int rowsB= blockDim.x/{{ THREADS_ROW }} ;
    unsigned int j=0;
    const int tid = threadIdx.x; // index in block
    if(row<shRows)
    { 
        const int idxR = tid/{{ THREADS_ROW }}; //row index mapped into block region
        const int idxT = tid%{{ THREADS_ROW }}; // thread number in Thread Group
    
        float preVals[{{ PREFETCH_SIZE }}];
        int preColls[{{ PREFETCH_SIZE }}];
    
        float dot[{{ PREFETCH_SIZE }}]={{ PREFETCH_INIT_TAB }};
    
        int maxEl = rowLength[row]; //original row length divided by T*PREFETCH

        unsigned int arIdx=0;
    
        for(int i=0; i<maxEl;i++)
        {        
            #pragma unroll
            for( j=0; j<{{ PREFETCH_SIZE }};j++)			
            {
                arIdx = (i*{{ PREFETCH_SIZE }}+j)*shRows*{{ THREADS_ROW }}+row*{{ THREADS_ROW }}+idxT;
                preColls[j]=colIdx[arIdx];
                preVals[j]=vals[arIdx];
            }
            
            #pragma unroll
            for( j=0; j<{{ PREFETCH_SIZE }};j++){
                dot[j]+=preVals[j]*tex1Dfetch(mainVecTexRef,preColls[j]);
            }
        }
        
        #pragma unroll
        for( j=1; j<{{ PREFETCH_SIZE }};j++){
            dot[0]+=dot[j];
        }

        // special indexing, values for example for T=4 BlockSize=256
        //for row=0 values are stored on position 0,64,128,192 
        //for row=1 values are stored on position 1,65,129,193 ...
        shDot[idxT*rowsB+idxR]=dot[0];
    
        __syncthreads();		
    }
    
    volatile float *shDotv = shDot;
    //reduction to some level
    for( j=blockDim.x/2; j>=rowsB; j>>=1) //s/=2
    {
        if(tid<j){
            shDotv[tid]+=shDotv[tid+j];
        }
        __syncthreads();
    }
    if(threadIdx.x<rowsB){		
        unsigned int row2=blockIdx.x* rowsB+threadIdx.x;
        if(row2<shRows)
            results[row2]=shDotv[threadIdx.x];
    }	
}