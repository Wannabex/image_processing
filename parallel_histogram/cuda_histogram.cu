#include <stdio.h>
#include <stdlib.h>
#include "bitmap.h"


#define MAXHISTVAL 0xff
#define NBLOCKS 4 
#define NTHREADS 10

__global__ void histogram(unsigned char *img, int* imgSize, unsigned int *histR, unsigned int *histG, unsigned int *histB) 
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ int sharedHistR[MAXHISTVAL + 1];
	__shared__ int sharedHistG[MAXHISTVAL + 1];
	__shared__ int sharedHistB[MAXHISTVAL + 1];

	
	int i = 0;
	for (i; i <= MAXHISTVAL; i++)
	{
		histR[i] = 0;		
		histG[i] = 0;		
		histB[i] = 0;			
		sharedHistR[i] = 0;		
		sharedHistG[i] = 0;		
		sharedHistB[i] = 0;			
	}
	
	__syncthreads();

	/*
	Every specific thread will check image byte starting at index = threadIdx.x + blockIdx.x * blockDim.x
	and then the next ones, each after NBLOCKS * NTHREADS more bytes untill the end is reached.
	*/

	int j = index;
	for(j; j < *imgSize; j += (NBLOCKS*NTHREADS))
	{
		unsigned char pixelClrVal = img[j];
		if(j % 3 == 0)
		{						
			atomicAdd(&sharedHistB[pixelClrVal], 1);
		//	atomicAdd(&histB[pixelClrVal], 1);			
		}
		else if(j % 3 == 1)
		{			
			atomicAdd(&sharedHistG[pixelClrVal], 1);
		//	atomicAdd(&histG[pixelClrVal], 1);			
		}
		else
		{						
			atomicAdd(&sharedHistR[pixelClrVal], 1);
		//	atomicAdd(&histR[pixelClrVal], 1);			
		}	
	}
	__syncthreads();	
	
	if (threadIdx.x == 0)
	{
		int k = 0;
		for (k; k <= MAXHISTVAL; k++)
		{	
			atomicAdd(&histR[k], sharedHistR[k]);
			atomicAdd(&histG[k], sharedHistG[k]);
			atomicAdd(&histB[k], sharedHistB[k]);					
		}	
	}
	__syncthreads();
	
}

/*
* host program
*/
int main(void) {
BITMAPHEADERS bitmapHeaders;
unsigned char* imgData;
int* sizeImgData;
unsigned int *histogramRed;
unsigned int *histogramGreen;
unsigned int *histogramBlue;
int sizeHistogram = (MAXHISTVAL + 1) * sizeof(int);
histogramRed = (unsigned int*) malloc(sizeHistogram);
histogramGreen = (unsigned int*) malloc(sizeHistogram);
histogramBlue = (unsigned int*) malloc(sizeHistogram);

imgData = loadBitmapFile("./img1.bmp", &bitmapHeaders);
if(imgData == NULL)
{
	printf("could not load bitmap file");
	return 1;
}
sizeImgData = &bitmapHeaders.infoHeader.biSizeImage;


 unsigned char* d_imgData;
 int* d_sizeImgData;
 unsigned int *d_histogramRed;
 unsigned int *d_histogramGreen;
 unsigned int *d_histogramBlue;

 int nBlk = NBLOCKS;
 int nThx = NTHREADS;
 //int N = nBlk * nThx;

 // Alloc space for device copies of histogram colors and img data
 cudaMalloc((void **)&d_imgData, *sizeImgData);
 cudaMalloc((void **)&d_sizeImgData, sizeof(int));
 cudaMalloc((void **)&d_histogramRed, sizeHistogram);
 cudaMalloc((void **)&d_histogramGreen, sizeHistogram);
 cudaMalloc((void **)&d_histogramBlue, sizeHistogram);

 // Copy input data to device
 cudaMemcpy(d_imgData, imgData, *sizeImgData, cudaMemcpyHostToDevice); 
 cudaMemcpy(d_sizeImgData, sizeImgData, sizeof(int), cudaMemcpyHostToDevice); 

 // Launch histogram() kernel on GPU with nBlk blocks each with nThx threads
 histogram<<<nBlk,nThx>>>(d_imgData, d_sizeImgData, d_histogramRed, d_histogramGreen, d_histogramBlue);


 // Copy result back to host 
 cudaMemcpy(histogramRed, d_histogramRed, sizeHistogram, cudaMemcpyDeviceToHost);
 cudaMemcpy(histogramGreen, d_histogramGreen, sizeHistogram, cudaMemcpyDeviceToHost);
 cudaMemcpy(histogramBlue, d_histogramBlue, sizeHistogram, cudaMemcpyDeviceToHost);


 printf("CUDA finished.\n Results are saved in the file \n");
 FILE* txtFile = fopen("./cuda_results.txt", "w+");	
 fprintf(txtFile, "Bin value: ");
 int i = 0;
 for(i; i <= MAXHISTVAL; i++)
 {
	fprintf(txtFile, "%u, ", i);
 }

 fprintf(txtFile, "\n Red hist: ");
 int j = 0;
 for(j; j <= MAXHISTVAL; j++)
 {
        fprintf(txtFile, "%u, ", histogramRed[j]);                
 }

 fprintf(txtFile, "\n Green hist:");
 int k = 0;
 for(k; k <= MAXHISTVAL; k++)
 {
        fprintf(txtFile, "%u, ", histogramGreen[k]);
 }

 fprintf(txtFile, "\n Blue hist:");
 int l = 0;
 for(l; l <= MAXHISTVAL; l++)
 {
        fprintf(txtFile, "%u, ", histogramBlue[l]);
 }
 fclose(txtFile);
 
 // Cleanup
 free(imgData); free(histogramRed); free(histogramGreen); free(histogramBlue);
 cudaFree(d_imgData); cudaFree(d_sizeImgData); cudaFree(d_histogramRed); cudaFree(d_histogramGreen); cudaFree(d_histogramBlue);
 return 0;
}
