//
// Created by 12111208 on 2022/11/23.
//
#include <stdio.h>
#include <stdbool.h>
#ifndef PROJECT_4_FUNCTION_H
#define PROJECT_4_FUNCTION_H
typedef struct martix{
    size_t rows;
    size_t cols;
    float *data;
} martix;
martix *creatRandomMatrix(size_t rows,size_t cols);
martix *creatZeroMatrix(size_t rows,size_t cols);
bool deleteMatrix( martix *origin);
void printMartix(const martix *origin);
martix *matmul_plain(const martix *m1, const martix *m2);
martix *matmul_improved(const martix *m1, const martix *m2);
bool judgeNull(const martix *origin);
martix *matmul_oblas(const martix *m1, const martix *m2);
bool *judgeEqual(const martix *m1, const martix *m2);
#endif //PROJECT_4_FUNCTION_H
