#include<omp.h>
#include<stdio.h>
#include <stdlib.h>
#include "bitmap.h"

#define MAXHISTVAL 0xff
#define NTHREADS 8

int main()
{
	BITMAPHEADERS bitmapHeaders;
	unsigned char* imgData;
	long imgDataLength;

	unsigned int *histogramRed;
	unsigned int *histogramGreen;
	unsigned int *histogramBlue;
	histogramRed = (unsigned int*) malloc(MAXHISTVAL * sizeof(int));
	histogramGreen = (unsigned int*) malloc(MAXHISTVAL * sizeof(int));
	histogramBlue = (unsigned int*) malloc(MAXHISTVAL * sizeof(int));

	memset(histogramRed, 0, MAXHISTVAL * sizeof(int));
	memset(histogramGreen, 0, MAXHISTVAL * sizeof(int));
	memset(histogramBlue, 0, MAXHISTVAL * sizeof(int));

	imgData = loadBitmapFile("./img1.bmp", &bitmapHeaders);
	if(imgData == NULL)
	{
		printf("could not load bitmap file");
		return 1;
	}
	imgDataLength = bitmapHeaders.infoHeader.biSizeImage;	

	
	omp_set_num_threads(NTHREADS);
	#pragma omp parallel  shared(histogramRed, histogramGreen, histogramBlue)
	{	
		unsigned int i = 0;		
		#pragma omp for
		for(i; i < imgDataLength; i++)	
		{
			if(i % 3 == 0)
			{				
				#pragma omp atomic
				histogramBlue[imgData[i]] += 1;
			}
			else if(i % 3 == 1)
			{				
				#pragma omp atomic
				histogramGreen[imgData[i]] += 1;
			}
			else
			{
				#pragma omp atomic
				histogramRed[imgData[i]] += 1;
			}				
		}	
	}	
	printf("OMP finished.\n Results are saved in the file \n");
	FILE* txtFile = fopen("./omp_results.txt", "w+");	

	fprintf(txtFile, "Bin value: ");
	int m = 0;
	for(m; m <= MAXHISTVAL; m++)
	{
		fprintf(txtFile, "%u, ", m);
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
	free(imgData); free(histogramRed); free(histogramGreen); free(histogramBlue);
	return 0;
}

