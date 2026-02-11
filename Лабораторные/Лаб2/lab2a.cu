#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

int *a, *b;


void h_Test(int n, int* a, int* b){
    int i;
    for(i = 0; i < n; i++){
        a[i] += b[i];
    }
}

int main()
{
    int vector_size = 1;
    printf("\nInput vector size: ");
    scanf("%d", &vector_size);

    struct timeval t;
    double start, finish;
    double elapsed;

    int n = vector_size;
    if (n == 0){
        n = 1 << 30;
    }
    
    a = (int*)calloc(n, sizeof(int));
    b = (int*)calloc(n, sizeof(int));

    for(i = 0; i < n; i++){
        a[i] = 2*i;
        b[i] = 2*i + 1;
    }


    gettimeofday(&t, NULL);
    start = (double)t.tv_sec*1000000.0 + (double)t.tv_usec;
    h_Test(n, a, b);
    gettimeofday(&t, NULL);
    finish = (double)t.tv_sec*1000000.0 + (double)t.tv_usec;
    elapsed = (double)(finish - start)/1000.0;

    printf("Elapsed time: %g ms\n", elapsed);

    free(Targs);
    free(th_id);

    for(i = 0; i < n; i++){
        printf("%d\t%d\t%d\t", i, b[i], a[i]);
    }

    return 0;
}