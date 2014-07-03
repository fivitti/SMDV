# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:02:08 2014

@author: HP
"""

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

def getSlicedELLCudaCode():
    return '''
        texture<float, 1, cudaReadModeElementType> mainVecTexRef;

        extern "C" __global__ void SlicedEllpackFormatKernel(
        	const float* vecVals,
        	const int* vecCols,
        	const int* vecLengths, 
        	const int* sliceStart, 
        	float* result,
        	const int nrRows, 
        	const int align,
        	const int ThreadPerRow)
        	{
                 __shared__  float sh_cache[100];
                 int tx = threadIdx.x;
                 int txm = tx %  ThreadPerRow;
        		int thIdx = (blockIdx.x*blockDim.x+tx);
        		int row = thIdx/ThreadPerRow;  //thIdx>> 2;
        
        		if (row < nrRows){
        			float sub = 0.0;
        			int maxRow = (int)ceil(vecLengths[row]/(float)ThreadPerRow);
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
           
                       for(int s=ThreadPerRow/2; s>0; s>>=1) //s/=2
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

def getSertlipCudaCode():
    return '''
texture<float, 1, cudaReadModeElementType> mainVecTexRef;
const int PREFETCH = 8;

extern "C" __global__ void rbfSERTILP_old(const float *vecVals,
	const int *vecCols,
	const int *vecLengths, 
	const int * sliceStart, 
	float *result,
	const int nrRows, 
	const int align,
	const int ThreadPerRow,
	const int SliceSize,
	const int prefetch_size){

		__shared__  float shDot[260];	
		__shared__ int shSliceStart;

		if(threadIdx.x==0)
		{
			shSliceStart=sliceStart[blockIdx.x];
		}
		__syncthreads();

		int idxT = threadIdx.x % ThreadPerRow; //thread number in Thread group
		int idxR = threadIdx.x/ThreadPerRow; //row index mapped into block region

		//map group of thread to row, in this case 4 threads are mapped to one row
		int row = (blockIdx.x*blockDim.x+threadIdx.x)/ThreadPerRow; //(blockIdx.x*blockDim.x+threadIdx.x)>> LOG_THREADS; 

		if (row < nrRows){
			int maxRow = vecLengths[row];
			//int maxRow = (int)ceil(vecLengths[row]/(float)(ThreadPerRow*prefetch_size) );

			float val[PREFETCH];
			int col[PREFETCH];
			float dot[PREFETCH]={0};

			unsigned int j=0;
			unsigned int arIdx=0;
			for(int i=0; i < maxRow; i++){

#pragma unroll
				for( j=0; j<prefetch_size;j++)	{
					arIdx = (i*prefetch_size+j )*align+shSliceStart+threadIdx.x;
					col[j] = vecCols[arIdx];
					val[j] = vecVals[arIdx];
				}

#pragma unroll
				for( j=0; j<prefetch_size;j++){
					dot[j]+=val[j]*tex1Dfetch(mainVecTexRef,col[j]); 
				}
			}

#pragma unroll
			for( j=1; j<prefetch_size;j++){
				dot[0]+=dot[j];	
			}



			shDot[idxT*SliceSize+idxR]=dot[0];
			__syncthreads();		

			volatile float *shDotv = shDot;
			//reduction to some level
			for( j=blockDim.x/2; j>=SliceSize; j>>=1) //s/=2
			{
				if(threadIdx.x<j){
					shDotv[threadIdx.x]+=shDotv[threadIdx.x+j];
				}
				__syncthreads();
			}

			if(threadIdx.x<SliceSize){			
				unsigned int row2=blockIdx.x* SliceSize+threadIdx.x;
				if(row2<nrRows){
					result[row2]= shDotv[threadIdx.x];
				}
			}

		}//if row<nrRows 
}//end func
    '''