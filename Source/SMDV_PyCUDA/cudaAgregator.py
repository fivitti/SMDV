# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:02:08 2014

@author: HP
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
           }///dodalem nawias zamykajacy
        
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
    
if __name__ == "__main__":
    p = getSertilpCudaCode()
    print p