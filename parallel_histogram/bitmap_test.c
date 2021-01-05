#include "bitmap.h"


int main()
{
	printf("starting program \n");
	unsigned char *bitmapData;
	BITMAPHEADERS bitmapHeaders;
	bitmapData = loadBitmapFile("./img1.bmp", &bitmapHeaders);
	if(bitmapData == NULL)
	{
		printf("Could not load bitmap file \n");
		return 1;
	}
	printf("%d \n", bitmapHeaders.fileHeader.dataOffset);
	printf("%d \n", bitmapHeaders.infoHeader.biSize);
	printf("img width %d \n", bitmapHeaders.infoHeader.biWidth);
	printf("img height %d \n", bitmapHeaders.infoHeader.biHeight);
	printf("img total size %d \n", bitmapHeaders.infoHeader.biSizeImage);
	printf("bits per pixel %d \n", bitmapHeaders.infoHeader.biBitCount);


	printf("ending program \n");
	return 0;
}
