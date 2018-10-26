
#include <stdio.h>
#include <math.h>
#include "mytype.h"

double tenegrad(u8 *grayMap, s32 width, s32 height, s32 lineByte)
{
	int i, j;
	u8 *bmpLine1;
	u8 *bmpLine2;
	u8 *bmpLine3;

	double S = 0;

    int Sx, Sy;
	if (grayMap == NULL) {
		return -1;
	}

	for (i = 0; i < height - 2; i++) {
		bmpLine1 = grayMap + i*lineByte;
		bmpLine2 = grayMap + (i + 1)*lineByte;
		bmpLine3 = grayMap + (i + 2)*lineByte;

		for (j = 0; j < width - 2; j++) {
            Sx = bmpLine1[j+2] + 2*bmpLine2[j+2] + bmpLine3[j+2]
                -bmpLine1[j]   - 2*bmpLine2[j]   - bmpLine3[j];
            Sy = bmpLine3[j] + 2*bmpLine3[j+1] + bmpLine3[j+2]
                -bmpLine1[j] - 2*bmpLine1[j+1] - bmpLine1[j+2];

			S += Sx*Sx + Sy*Sy;
		}
	}

    S = S/(width-2)/(height-2);
    return S;
}
//������˹�Ľ�������
double laplaceAdvance(u8 *grayMap, s32 width, s32 height, s32 lineByte)
{
	int i, j;
	u8 *bmpLine1;
	u8 *bmpLine2;
	u8 *bmpLine3;

	s32 l1;
	s32 l2;
	s32 l3;

	s64 llvalue = 0;

	if (grayMap == NULL) {
		return -1;
	}

	for (i = 0; i < height - 2; i++) {
		bmpLine1 = grayMap + i*lineByte;
		bmpLine2 = grayMap + (i + 1)*lineByte;
		bmpLine3 = grayMap + (i + 2)*lineByte;

		for (j = 0; j < width - 2; j++) {

			l1 = bmpLine1[j + 2];
			l2 = 10 * bmpLine2[j + 1] - 4 * bmpLine2[j + 2];
			l3 = 4 * bmpLine3[j + 1] + bmpLine3[j + 2];
			llvalue = llvalue + (l2 - l1 - l3)*(l2 - l1 - l3);
		}
	}

	return (double)llvalue;
}

//laplace����
double laplace(u8 *grayMap, s32 width, s32 height, s32 lineByte)
{
	int i, j;
	u8 *bmpLine1;
	u8 *bmpLine2;
	u8 *bmpLine3;

	s32 l1;
	s32 l2;
	s32 l3;

	s64 llvalue = 0;

	if (grayMap == NULL) {
		return -1;
	}

	for (i = 0; i < height - 2; i++) {
		bmpLine1 = grayMap + i*lineByte;
		bmpLine2 = grayMap + (i + 1)*lineByte;
		bmpLine3 = grayMap + (i + 2)*lineByte;

		for (j = 0; j < width - 2; j++) {

			l1 = bmpLine1[j] + 4 * bmpLine1[j+1] + bmpLine1[j+2];
			l2 = 20 * bmpLine2[j + 1] - 4 * bmpLine2[j + 2] - 4 * bmpLine2[j];
			l3 = bmpLine3[j] + 4 * bmpLine3[j + 1] + bmpLine3[j + 2];
			llvalue = llvalue + (l2-l1-l3)*(l2-l1-l3);
		}
	}

	return (double)llvalue;
}

//�����ݶ�����
double energyVariance(u8 *grayMap, s32 width, s32 height, s32 lineByte)
{
	int i, j;
	u8 *bmpBuffer = NULL;
	u8 *bmpBufferNext = NULL;

	s64 variance = 0;
	if (grayMap == NULL) {
		return -1;
	}

	for (i = 0; i < height-1; i++) {

		bmpBuffer = grayMap + i*lineByte; 
		bmpBufferNext = grayMap + (i + 1) * lineByte;

		for (j = 0; j < width - 1; j++) {
			variance = variance + 
				((bmpBuffer[j + 1] - bmpBuffer[j]) * (bmpBuffer[j + 1] - bmpBuffer[j])) + 
				((bmpBufferNext[j] - bmpBuffer[j]) * (bmpBufferNext[j] - bmpBuffer[j]));
		}
	}

	return (double)variance;
}

//brenner����
double grayBrenner(u8 *grayMap, s32 width, s32 height, s32 lineByte)
{
	int i, j;
	u8 *bmpBuffer = NULL;

	s64 brenner = 0;

	for (i = 0; i < height; i++) {
		bmpBuffer = grayMap + i*lineByte;
		for (j = 2; j < width; j++) {
			brenner = brenner + ((bmpBuffer[j] - bmpBuffer[j - 2]) * (bmpBuffer[j] - bmpBuffer[j - 2]));
		}	
	}

	return (double)brenner;
}

//�Ҷȷ�������
double grayVariance(u8 *grayMap, s32 width, s32 height, s32 lineByte)
{
	int i, j;
	u8 *bmpBuffer = NULL;

	s64 energySum = 0;

	double energy = 0.0;
	double wh = 0.0;
	double variance = 0.0;

	if (grayMap == NULL) {
		return -1;
	}

	//����ͼ��Ҷȵ�ƽ��ֵ
	for (i = 0; i < height; i++)
	{
		bmpBuffer = grayMap + i*lineByte;
		for (j = 0; j < width; j++) {
			energySum += bmpBuffer[j];
		}
	}
	wh = (double) (width * height);
	energy = (double) energySum;
	energy = energy / wh;

	//���㷽��
	for (i = 0; i < height; i++) {
		bmpBuffer = grayMap + i*lineByte;
		for (j = 0; j < width; j++) {
			variance = variance + (bmpBuffer[j] - energy) * (bmpBuffer[j] * energy);
		}
	}

	//*grayVariance = variance / wh;
	return variance;
}

//����ͼ���һά��
double grayEntropy(u8 *grayMap, s32 width, s32 height, s32 lineByte)
{
	int i, j;

	u8 pixelGrayValue; //���ػҶ�ֵ
	u8 *bmpBuffer = NULL;
	
	double pi[256];
	double entropy = 0.0;
	double wh = 0.0;

	if (grayMap == NULL) {
		return -1;
	}

	//����Ҷȳ��ָ��ʱ�
	for (i = 0; i < 256; i++) {
		pi[i] = 0.0;
	}

	//����ÿ�����س��ֵĴ���
	for (i = 0; i < height; i++) {
		bmpBuffer = grayMap + i*lineByte;

		for (j = 0; j < width; j++) {
			//i[bmpBuffer[j]] = 
			pixelGrayValue = bmpBuffer[j];
			pi[pixelGrayValue] = pi[pixelGrayValue] + 1;
		}
	}

	//����ÿ��������ͼ����ֵĸ���
	wh = (double)width*height;
	for (i = 0; i < 256; i++) {
		pi[i] = pi[i] / wh;	
	}

	// ���ݶ������ͼ����
	entropy = 0.0;
	for (i = 0; i < 256; i++) {
		if (pi[i] != 0.0) {
			entropy = entropy - pi[i] * (log(pi[i]) / log(2.0));
		}
	}

	return entropy;
}


