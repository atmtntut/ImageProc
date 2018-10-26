#ifndef _MY_TYPE_H_
#define _MY_TYPE_H_

typedef union {
	struct {
		unsigned int b : 8;
		unsigned int g : 8;
		unsigned int r : 8;
		unsigned int a : 8;
	} RGB;

	unsigned int 	value32;
}Pixel64;

typedef union {
	struct {
		unsigned char b;
		unsigned char g;
		unsigned char r;

	}RGB;
	unsigned char pixel[3];
}Pixel24;


//整数类型定义
typedef unsigned int        u32;
typedef unsigned short      u16;
typedef unsigned char       u8;
typedef unsigned long long  u64;

typedef int                 s32;
typedef short               s16;
typedef char                s8;
typedef long long           s64;

typedef void *             HANDLE;


//位置信息定义（x，y）
typedef struct tagPosition {
	s32 x;
	s32 y;
}position;

//显示位置区域
typedef struct tagRegion {
	s32 x;
	s32 y;
	s32 w;
	s32 h;
}region;

typedef struct tag3DPosition {
	s32 x;
	s32 y;
	s32 z;
}position3D;

typedef struct tagBitmapInfo {

	s32 width;
	s32 height;
	s32 lineBytes;
	s32 bitCount;

}bitmapInfo;

#endif
