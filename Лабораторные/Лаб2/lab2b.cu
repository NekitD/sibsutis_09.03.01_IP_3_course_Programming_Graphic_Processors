#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <pthread.h>

int *a, *b;

struct targ{
    int num_thread,
    int num_threads,
    int length
}

void* h_Test(void* arg){
    struct targ* s_arg = (struct targ*)arg;
    int length = s_arg->length;
    int offset = s_arg->num_thread*length;
    int i;
    for(i = 0; i < length; i++){
        a[i+offset] += b[i+offset];
    }
    return NULL
}

int main()
{
    char prog_name[50] = "";
    int num_of_threads = 1, vector_size = 1;
    printf("Input program name: ");
    scanf("%s", &prog_name);
    printf("\nInput num of threads: ");
    scanf("%d", &num_of_threads);
    printf("\nInput vector size: ");
    scanf("%d", &vector_size);

    struct timeval t;
    double start, finish;
    double elapsed;

    int i;
    int th_n = num_of_threads;
    int n = vector_size;

    struct targ* Targs = (struct targ*)calloc(th_n, sizeof(struct targ*));
    pthread_t* th_id = (pthread_t*)calloc(th_n, sizeof(pthread_t));
    
    a = (int*)calloc(n, sizeof(int));
    b = (int*)calloc(n, sizeof(int));

    for(i = 0; i < n; i++){
        a[i] = 2*i;
        b[i] = 2*i + 1;
    }

    for(i = 0; i < th_n; i++){
        Targs[i].num_threads = th_n;
        Targs[i].num_thread = i;
        Targs[i].length = n/th_n;
    }

    gettimeofday(&t, NULL);
    start = (double)t.tv_sec*1000000.0 + (double)t.tv_usec;
    for(i = 0; i < th_n; i++){
        pthread_create(&th_id[i], NULL, &hTest, &Targs[i]);
    }
    for(i = 0; i < th_n; i++){
        pthread_join(th_id[i], NULL);
    }
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