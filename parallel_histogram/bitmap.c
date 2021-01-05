// bitmap.c
// structurec and function to read .bmp image file

#include <stdio.h>
#include <stdlib.h>
#include "bitmap.h"


unsigned char* loadBitmapFile(char* filename, BITMAPHEADERS* bitmapHeaders)// BITMAPINFOHEADER* bitmapInfoHeader)
{
	FILE* img = fopen(filename, "rb");
	if (img	== NULL)
	{
		return NULL;
	}		
	unsigned char* bitmapData;
	
	// first read bitmap file header
	fread(&(bitmapHeaders->fileHeader), sizeof(BITMAPFILEHEADER), 1, img);
	// fread does indeed move FILE pointer!!
	if(bitmapHeaders->fileHeader.type != 0x4D42) // check if file is for sure of bmp format
	{
		fclose(img);
		return NULL;
	}
	
	// read bitmap info header
	fread(&(bitmapHeaders->infoHeader), sizeof(BITMAPINFOHEADER), 1, img);

	// moving FILE pointer at the beggning of data. SEEK_SET to move relative to position 0 in the file
	fseek(img, bitmapHeaders->fileHeader.dataOffset, SEEK_SET);

	bitmapData = (unsigned char*)malloc(bitmapHeaders->infoHeader.biSizeImage);

	if(!bitmapData)
	{
		free(bitmapData);
		fclose(img);
		return NULL;
	}

	// at last, read pixels data into bytes array;
	fread(bitmapData, sizeof(char), bitmapHeaders->infoHeader.biSizeImage, img);
	// bitmap data is BGR not RGB!!

	fclose(img);
	return bitmapData;	
}


