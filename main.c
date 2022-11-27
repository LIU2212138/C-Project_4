#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <stdio.h>
#include <malloc.h>
#include "function.h"
#include <time.h>

int main() {
    struct martix *a=creatRandomMatrix(8000,8000);
    
    struct martix *b=creatRandomMatrix(8000,8000);

    double startq = omp_get_wtime( );
    struct martix* c = matmul_plain(a,b);
    double endq = omp_get_wtime( );
    
    double starto = omp_get_wtime( );
    struct martix* e = matmul_oblas(a,b);
    double endo = omp_get_wtime( );

    double start = omp_get_wtime( );
    struct martix* d = matmul_improved(a,b);
    double end = omp_get_wtime( );

    


    printf("----------------The running time of normal method is : %f(seconds)-----------\n",endq-startq);
    printf("----------------The running time of improved method is : %f(seconds)-----------\n",end-start);
    printf("----------------The running time of openblas is : %f(seconds)-----------\n",endo-starto);
    printf("----------------Are c,d are equal? %d-----------\n",(int )judgeEqual(c,d));
    printf("----------------Are d,e are equal? %d-----------\n",judgeEqual(d,e));
    deleteMatrix(a);
    deleteMatrix(b);
     deleteMatrix(c);
    deleteMatrix(d);
    deleteMatrix(e);
    return 0;

    }


