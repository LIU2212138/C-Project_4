//
// Created by 12111208 on 2022/11/23.
//

#include <malloc.h>
#include "function.h"
#include <time.h>
#include <string.h>
#include <immintrin.h>
#include <omp.h>
#include <cblas.h>

martix *creatRandomMatrix(size_t rows,size_t cols){
    if(rows==0||cols==0){
        fprintf(stderr,"ERROR:File %s, Line %d, Function %s(): Invalid size of matrix:row is %zu, col is %zu.\n",__FILE__,__LINE__,__FUNCTION__,rows,cols);
    }
    martix *origin= (martix *) malloc(sizeof (martix));
    if (origin==NULL){
        fprintf(stderr,"ERROR:File %s, Line %d, Function %s(): Memory request failed.\n",__FILE__,__LINE__,__FUNCTION__);
    }
    origin->rows = rows;
    origin->cols = cols;
    origin->data = (float *) malloc(*(&origin->rows) * *(&origin->cols) * sizeof(float)+20);
    if(origin->data != NULL){
        srand((unsigned) time(NULL));
        for (int i = 0; i < origin->rows; ++i) {
            for (int j = 0; j < origin->cols; ++j) {
                origin->data[i*origin->cols+j]=rand()%10+1;
            }
        }
        return origin;
    }else {
        fprintf(stderr,"ERROR:File %s, Line %d, Function %s(): Memory request failed.\n",__FILE__,__LINE__,__FUNCTION__);
        free(origin);
        return NULL;
    }
}
martix *creatZeroMatrix(size_t rows,size_t cols){
    if(rows==0||cols==0){
        fprintf(stderr,"ERROR:File %s, Line %d, Function %s(): Invalid size of matrix:row is %zu, col is %zu.\n",__FILE__,__LINE__,__FUNCTION__,rows,cols);
    }
    martix *origin= (martix *) malloc(sizeof (martix));
    if(origin==NULL){
        fprintf(stderr,"ERROR:File %s, Line %d, Function %s(): Memory request failed.\n",__FILE__,__LINE__,__FUNCTION__);
    }
    origin->rows = rows;
    origin->cols = cols;
    origin->data = (float *) malloc(*(&origin->rows) * *(&origin->cols) * sizeof(float)+20);
    if(origin->data != NULL){
        for (int i = 0; i < origin->rows; ++i) {
            for (int j = 0; j < origin->cols; ++j) {
                origin->data[i*origin->rows+j]=0;
            }
        }
        return origin;
    }else {
        fprintf(stderr,"ERROR:File %s, Line %d, Function %s(): Memory request failed.\n",__FILE__,__LINE__,__FUNCTION__);
        free(origin);
        return NULL;
    }
}
void printMartix(const martix *origin){
    if(judgeNull(origin)){
        return;
    }
    for (int i = 0; i < origin->rows; ++i) {
        for (int j = 0; j < origin->cols; ++j) {
            printf("%f     ",*(&origin->data[i*origin->cols+j]));
        }
        printf("\n");
    }
}
bool deleteMatrix(martix *origin){
    if(judgeNull(origin)){
        return false;
    }
    //困难：一直遇见断点陷阱：解决：malloc多分点内存
    free((origin)->data);
    (origin)->data=NULL;
    free(origin);
    origin=NULL;
    return true;
}
martix *matmul_plain(const martix *m1,const martix *m2){
    if(judgeNull(m1)){
        fprintf(stderr,"ERROR:File %s, Line %d, Function %s(): m1 is null.\n",__FILE__,__LINE__,__FUNCTION__);
        return NULL;
    }
    if (judgeNull(m2)){
        fprintf(stderr,"ERROR:File %s, Line %d, Function %s(): m2 is null.\n",__FILE__,__LINE__,__FUNCTION__);
        return NULL;
    }
    if (m1->cols!=m2->rows){
        fprintf(stderr,"ERROR:File %s, Line %d, Function %s(): the length of m1's column is not equal to m2's row.\n",__FILE__,__LINE__,__FUNCTION__);
        return NULL;
    }
    martix *p = creatZeroMatrix(m1->rows,m2->rows);//注意一下这个初始化， 不初始化为0的话会使普通方法得出来的数被内存之前存的数干扰

    for (int i = 0; i < p->rows; ++i) {
        for (int j = 0; j < p->cols; ++j) {
            for (int k = 0; k < m2->rows; ++k) {
                p->data[i*p->cols+j]+=m1->data[i*m1->cols+k]*m2->data[k*m2->cols+j];
            }
        }
    }
    return p;
}
martix *matmul_improved(const martix *m1, const martix *m2){
    if(judgeNull(m1)|| judgeNull((m2))||m1->cols!=m2->rows){
        fprintf(stderr,"ERROR:File %s, Line %d, Function %s(): m1 is null.\n",__FILE__,__LINE__,__FUNCTION__);
        return NULL;
    }
    if (judgeNull(m2)){
        fprintf(stderr,"ERROR:File %s, Line %d, Function %s(): m2 is null.\n",__FILE__,__LINE__,__FUNCTION__);
        return NULL;
    }
    if (m1->cols!=m2->rows){
        fprintf(stderr,"ERROR:File %s, Line %d, Function %s(): the length of m1's column is not equal to m2's row.\n",__FILE__,__LINE__,__FUNCTION__);
        return NULL;
    }
    martix * result=creatZeroMatrix(m1->rows,m2->cols);

    unsigned long long len = m1->cols/8;

#pragma omp parallel for
    for (size_t i = 0; i < result->rows; ++i) {
        for (size_t j = 0; j < result->cols; ++j) {
            float sum = 0;
            float helper[8]={0,0,0,0,0,0,0,0};

            __m256 helperM = _mm256_setzero_ps();
            for (int k = 0; k < len; ++k) {
                //用loadu就不要求32位对齐，而load要求32位对齐
                __m256 m1Row = _mm256_loadu_ps(&(m1->data[m1->cols*i + k * 8]) );
                //第一次反了！！！！
                __m256 m2Col = _mm256_set_ps(m2->data[m2->cols*(k*8+7)+j],m2->data[m2->cols*(k*8+6)+j],m2->data[m2->cols*(k*8+5)+j],m2->data[m2->cols*(k*8+4)+j],m2->data[m2->cols*(k*8+3)+j],m2->data[m2->cols*(k*8+2)+j],m2->data[m2->cols*(k*8+1)+j],m2->data[m2->cols*(k*8+0)+j]);

                helperM = _mm256_fmadd_ps(m1Row,m2Col,helperM);
            }
            //把sum的循环相加改成了用fmadd先进行叠加，然后再存到sum中累加
            _mm256_storeu_ps(helper,helperM);
            sum += helper[0]+helper[1]+helper[2]+helper[3]+helper[4]+helper[5]+helper[6]+helper[7];
            int reminder = m1->cols%8;
            for (int k = 0; k < reminder; ++k) {
                sum+=m1->data[m1->cols*i + len*8+k]*m2->data[m2->cols*(len*8+k)+j];
            }
            result->data[result->cols*i+j]=sum;
        }
    }
    return result;

}
martix *matmul_oblas(const martix *m1, const martix *m2){
    if(judgeNull(m1)|| judgeNull((m2))||m1->cols!=m2->rows){
        fprintf(stderr,"ERROR:File %s, Line %d, Function %s(): m1 is null.\n",__FILE__,__LINE__,__FUNCTION__);
        return NULL;
    }
    if (judgeNull(m2)){
        fprintf(stderr,"ERROR:File %s, Line %d, Function %s(): m2 is null.\n",__FILE__,__LINE__,__FUNCTION__);
        return NULL;
    }
    if (m1->cols!=m2->rows){
        fprintf(stderr,"ERROR:File %s, Line %d, Function %s(): the length of m1's column is not equal to m2's row.\n",__FILE__,__LINE__,__FUNCTION__);
        return NULL;
    }
    martix * result=creatZeroMatrix(m1->rows,m2->cols);
    const int M = m1->rows;
    const int N = m2->cols;
    const int K = m1->cols;
    const double alpha = 1.0;
    const double beta = 0.0;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, m1->data, K, m2->data, N, beta, result->data, N);
    return result;
}

bool judgeNull(const martix *origin){
    if(origin==NULL){
        fprintf(stderr,"ERROR:File %s, Line %d, Function %s(): Null pointer,function exit with invalid return value\n",__FILE__,__LINE__,__FUNCTION__);
        return true;
    } else{
        return false;
    }
}
bool *judgeEqual(const martix *m1, const martix *m2){
    if(judgeNull(m1)|| judgeNull((m2))||m1->cols!=m2->rows){
        fprintf(stderr,"ERROR:File %s, Line %d, Function %s(): m1 is null.\n",__FILE__,__LINE__,__FUNCTION__);
        return false;
    }
    if (judgeNull(m2)){
        fprintf(stderr,"ERROR:File %s, Line %d, Function %s(): m2 is null.\n",__FILE__,__LINE__,__FUNCTION__);
        return false;
    }
    if (m1->rows!=m2->rows){
        fprintf(stderr,"ERROR:File %s, Line %d, Function %s(): the length of m1's row is not equal to m2's row.\n",__FILE__,__LINE__,__FUNCTION__);
        return false;
    }
    if (m1->cols!=m2->cols){
        fprintf(stderr,"ERROR:File %s, Line %d, Function %s(): the length of m1's row is not equal to m2's row.\n",__FILE__,__LINE__,__FUNCTION__);
        return false;
    }
    for (size_t i = 0; i < m1->rows*m1->cols; i++)
    {
        if (m1->data[i]!=m2->data[i])
        {
            return false;
        }
        
    }
    return true;
    
}
