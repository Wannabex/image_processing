#ifndef BITMAP_H
#define BITMAP_h

#include <stdio.h>
#include <stdlib.h>


#pragma pack(push, 1)

typedef struct bmpFileHeader
{
	short type;
	int size;
	short reserved1;
	short reserved2;
	int dataOffset;
}BITMAPFILEHEADER;

typedef struct bmpInfoHeader
{
	int biSize;
	int biWidth;
	int biHeight;
	short biPlanes;
	short biBitCount;
	int biCompression;
	int biSizeImage;
	int biXPelsPerMeter;
	int biYPelsPerMeter;
	int biClrUsed;
	char biClrImportant;
	char biClrRotation;
	short biReserved;
} BITMAPINFOHEADER;

typedef struct bitmapHeaders
{
	BITMAPFILEHEADER fileHeader;
	BITMAPINFOHEADER infoHeader;	
}BITMAPHEADERS;


#pragma pack(pop)

unsigned char* loadBitmapFile(char* filename, BITMAPHEADERS* bitmapHeaders);// BITMAPINFOHEADER* bitmapInfoHeader);



#endif //BITMAP_H
