# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:02:08 2014

@author: Sławomir Figiel
"""
def convertString(string, **kwargs):
    s = string
    for name, value in kwargs.items():
        value = str(value)
        s = s.replace("{{"+name+"}}", value)
        s = s.replace("{{ "+name+"}}", value)
        s = s.replace("{{"+name+" }}", value)
        s = s.replace("{{ "+name+" }}", value)
    return s
        
def getELLCudaCode():
    return '''
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
    '''

def getSlicedELLCudaCode(sh_cache_size, threadPerRow = 2):
    tpl = '''
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
        
        __global__ void copy_texture_kernel(float * data) {
           int ty=threadIdx.x;
           data[ty] =tex1Dfetch(mainVecTexRef, ty);
        }
        '''
#    tpl = tpl.replace("{{ sh_cache_size }}", str(sh_cache_size))
#    tpl = tpl.replace("{{ threadPerRow }}", str(threadPerRow))
    tpl = convertString(tpl, sh_cache_size=sh_cache_size, threadPerRow=threadPerRow)
    return tpl

def getSertilpCudaCode(shDot_size = 0, threadPerRow = 2, sliceSize = 32, prefetch = 2):
    if shDot_size == 0:
        shDot_size = threadPerRow * sliceSize
    tpl = '''
        texture<float, 1, cudaReadModeElementType> mainVecTexRef;
        
        extern "C" __global__ void rbfSERTILP_old(const float *vecVals,
        	const int *vecCols,
        	const int *vecLengths, 
        	const int * sliceStart, 
        	float *result,
        	const int nrRows, 
        	const int align){
        
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
                 //(blockIdx.x*blockDim.x+threadIdx.x)>> LOG_THREADS; 
        
        
                //to zmienilem, wyciagnalem wyzej
                 unsigned int j=0;
        		if (row < nrRows){
        			int maxRow = vecLengths[row];
        			//int maxRow = (int)ceil(vecLengths[row]/(float)({{ threadPerRow }}*{{ prefetch }}) );
        
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
                   }///dodalem nawias zamykajacy dla if
        
        			volatile float *shDotv = shDot;
        			//reduction to some level
        			for( j=blockDim.x/2; j>={{ sliceSize }}; j>>=1) //s/=2
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
        
                //zakomentowalem ponizej nawias zamykajacy
        		//}//if row<nrRows 
        }//end func
    '''
#    tpl = tpl.replace("{{ shDot_size }}", str(shDot_size))
#    tpl = tpl.replace("{{ threadPerRow }}", str(threadPerRow))
#    tpl = tpl.replace("{{ sliceSize }}", str(sliceSize))
#    tpl = tpl.replace("{{ prefetch }}", str(prefetch))
    tpl = convertString(tpl, shDot_size = shDot_size, threadPerRow = threadPerRow, sliceSize = sliceSize, prefetch = prefetch)
    
    return tpl

def getErtilpCudaCode_old(block_sice, threadPerRow, prefetch):
    tpl = '''
        texture<float,1,cudaReadModeElementType> labelsTexRef;

        __device__ void SpMV_ERTILP(const float * vals,
        									   const int * colIdx, 
        									   const int * rowLength,
        									   const int row,
        									   const int rowsB,
        									   const int shRows,
        									   volatile float* shDot)
        {
        
        	const int tid = threadIdx.x; // index in block
        	const int idxR = tid/{{ THREADS_ROW }}; //row index mapped into block region
        	const int idxT = tid%{{ THREADS_ROW }}; // thread number in Thread Group
        
        
        
        	float preVals[{{ PREFETCH_SIZE }}];
        	int preColls[{{ PREFETCH_SIZE }}];
        
        	float dot[{{ PREFETCH_SIZE }}]={0};
        
        	int maxEl = rowLength[row]; //original row length divided by T*PREFETCH
        
        	unsigned int j=0;
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
        			dot[j]+=preVals[j]*tex1Dfetch(labelsTexRef,preColls[j]);
        		}
        	}
        	
        	#pragma unroll
        	for( j=1; j<{{ PREFETCH_SIZE }};j++){
        		dot[0]+=dot[j];
        	}
        
        	//__syncthreads();	
        
        	// special indexing, values for example for T=4 BlockSize=256
        	//for row=0 values are stored on position 0,64,128,192 
        	//for row=1 values are stored on position 1,65,129,193 ...
        	shDot[idxT*rowsB+idxR]=dot[0];
        
        	__syncthreads();		
        
        	//reduction to some level
        	for( j=blockDim.x/2; j>=rowsB; j>>=1) //s/=2
        	{
        		if(tid<j){
        			shDot[tid]+=shDot[tid+j];
        		}
        		__syncthreads();
        	}
        
        
        }
        
        extern "C" __global__ void rbfERTILP(const float * vals,
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
        
        
        	//const int idx  = blockDim.x * blockIdx.x + threadIdx.x;  // global thread index
        	int row  = (blockDim.x * blockIdx.x + threadIdx.x)/{{ THREADS_ROW }};
        
        	const int rowsB= blockDim.x/{{ THREADS_ROW }} ;//{{ BLOCK_SIZE }}/{{ THREADS_ROW }};  //rows in block
        	//#define rowsB {{ BLOCK_SIZE }}/{{ THREADS_ROW }}
        
        	if(row<shRows)
        	{
        
        		SpMV_ERTILP(vals,colIdx,rowLength,row,rowsB,shRows,shDot);
        		//if(row2<shRows){
        		if(threadIdx.x<rowsB){
        			//results[row2]=row2;			
        			unsigned int row2=blockIdx.x* rowsB+threadIdx.x;
         			if(row2<shRows)
        				results[row2]=shDot[threadIdx.x];
        		}
        	}//if row<nrRows	
        
        }
        '''
    tpl = convertString(tpl, BLOCK_SIZE = block_sice, THREADS_ROW = threadPerRow, PREFETCH_SIZE = prefetch)
    return tpl
    
def getErtilpCudaCode(block_sice, threadPerRow, prefetch):
    tpl = '''
        texture<float,1,cudaReadModeElementType> labelsTexRef;
        
        extern "C" __global__ void rbfERTILP(const float * vals,
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
        
        
        	//const int idx  = blockDim.x * blockIdx.x + threadIdx.x;  // global thread index
        	int row  = (blockDim.x * blockIdx.x + threadIdx.x)/{{ THREADS_ROW }};
        
        	const int rowsB= blockDim.x/{{ THREADS_ROW }} ;//{{ BLOCK_SIZE }}/{{ THREADS_ROW }};  //rows in block
        	//#define rowsB {{ BLOCK_SIZE }}/{{ THREADS_ROW }}
			unsigned int j=0;
			const int tid = threadIdx.x; // index in block
        	if(row<shRows)
        	{
        
				
				const int idxR = tid/{{ THREADS_ROW }}; //row index mapped into block region
				const int idxT = tid%{{ THREADS_ROW }}; // thread number in Thread Group
			
				float preVals[{{ PREFETCH_SIZE }}];
				int preColls[{{ PREFETCH_SIZE }}];
			
				float dot[{{ PREFETCH_SIZE }}]={0};
			
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
						dot[j]+=preVals[j]*tex1Dfetch(labelsTexRef,preColls[j]);
					}
				}
				
				#pragma unroll
				for( j=1; j<{{ PREFETCH_SIZE }};j++){
					dot[0]+=dot[j];
				}
			
				//__syncthreads();	
			
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
			//if(row2<shRows){
			if(threadIdx.x<rowsB){
				//results[row2]=row2;			
				unsigned int row2=blockIdx.x* rowsB+threadIdx.x;
				if(row2<shRows)
					results[row2]=shDotv[threadIdx.x];
			}	
        
        }
        '''
    tpl = convertString(tpl, BLOCK_SIZE = block_sice, THREADS_ROW = threadPerRow, PREFETCH_SIZE = prefetch)
    return tpl
if __name__ == "__main__":
    p = getSertilpCudaCode()
    print p