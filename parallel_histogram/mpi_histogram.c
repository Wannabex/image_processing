#include <mpi.h>
#include "bitmap.h"

#define MAXHISTVAL 0xff

void computeHistogram(unsigned char* imagePart, int imagePartLen, int* histR, int* histG, int* histB);

main(int argc, char** argv){
int myRank;
int nProcesses;
int root;

BITMAPHEADERS bitmapHeaders;
unsigned char* imgDataFull;
unsigned char* imgDataPart;
int imgFullLength;
int imgPartLength;

unsigned int *histogramRed;
unsigned int *histogramGreen;
unsigned int *histogramBlue;
histogramRed = (unsigned int*) malloc(MAXHISTVAL * sizeof(int));
histogramGreen = (unsigned int*) malloc(MAXHISTVAL * sizeof(int));
histogramBlue = (unsigned int*) malloc(MAXHISTVAL * sizeof(int));
memset(histogramRed, 0, MAXHISTVAL * sizeof(int));
memset(histogramGreen, 0, MAXHISTVAL * sizeof(int));
memset(histogramBlue, 0, MAXHISTVAL * sizeof(int));
unsigned int *finalHistogramRed;
unsigned int *finalHistogramGreen;
unsigned int *finalHistogramBlue;
finalHistogramRed = (unsigned int*) malloc(MAXHISTVAL * sizeof(int));
finalHistogramGreen = (unsigned int*) malloc(MAXHISTVAL * sizeof(int));
finalHistogramBlue = (unsigned int*) malloc(MAXHISTVAL * sizeof(int));
memset(finalHistogramRed, 0, MAXHISTVAL * sizeof(int));
memset(finalHistogramGreen, 0, MAXHISTVAL * sizeof(int));
memset(finalHistogramBlue, 0, MAXHISTVAL * sizeof(int));

	MPI_Init( &argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);
	root = 0;

	if (myRank == root) 
	{	
		imgDataFull = loadBitmapFile("./img1.bmp", &bitmapHeaders);
		if(imgDataFull == NULL)
		{
			printf("could not load bitmap file");
			return 1;
		}
	}
	MPI_Bcast(&bitmapHeaders, sizeof(BITMAPHEADERS), MPI_CHAR, root, MPI_COMM_WORLD);
	imgFullLength = bitmapHeaders.infoHeader.biSizeImage;

	imgPartLength = imgFullLength / nProcesses;
	imgDataPart = (unsigned char*) malloc(imgPartLength * sizeof(char));

	
	MPI_Scatter(imgDataFull, imgPartLength, MPI_CHAR,
				imgDataPart, imgPartLength, MPI_CHAR,
				root, MPI_COMM_WORLD);

	computeHistogram(imgDataPart, imgPartLength, histogramRed, histogramGreen, histogramBlue);
	
	int j = 0;
	for(j; j <= MAXHISTVAL; j++)
	{
		MPI_Reduce(&histogramRed[j], &finalHistogramRed[j], 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);
		MPI_Reduce(&histogramGreen[j], &finalHistogramGreen[j], 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);
		MPI_Reduce(&histogramBlue[j], &finalHistogramBlue[j], 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);
	}

	MPI_Gather(imgDataPart, imgPartLength, MPI_CHAR,
				imgDataFull, imgPartLength, MPI_CHAR,
				root, MPI_COMM_WORLD);

	if (myRank == root)
	{
		printf("MPI finished.\n Results are saved in the file \n");
		FILE* txtFile = fopen("./mpi_results.txt", "w+");	

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
                        fprintf(txtFile, "%u, ", finalHistogramRed[j]);                
                }

		fprintf(txtFile, "\n Green hist:");
		int k = 0;
                for(k; k <= MAXHISTVAL; k++)
                {
                        fprintf(txtFile, "%u, ", finalHistogramGreen[k]);
                }

		fprintf(txtFile, "\n Blue hist:");
		int l = 0;
		for(l; l <= MAXHISTVAL; l++)
                {
                        fprintf(txtFile, "%u, ", finalHistogramBlue[l]);
                }
		fclose(txtFile);
	}		
	//free(imgDataFull); free(imgDataPart); free(histogramRed); free(histogramGreen); free(histogramBlue);
	//free(finalHistogramRed); free(finalHistogramGreen); free(finalHistogramBlue);
	MPI_Finalize();
}

void computeHistogram(unsigned char* imagePart, int imagePartLen, int* histR, int* histG, int* histB)
{	
	int i = 0;
	for(i; i < imagePartLen; i++)	
	{
		unsigned char pixelClrVal = imagePart[i];
		if(i % 3 == 0)
		{
			//red part of pixel
			histB[pixelClrVal] += 1;
		}
		else if(i % 3 == 1)
		{
			//green part of pixel
			histG[pixelClrVal] += 1;
		}
		else
		{
			//blue part of pixel
			histR[pixelClrVal] += 1;
		}	
	}
}

