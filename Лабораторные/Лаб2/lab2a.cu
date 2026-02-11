#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <pthread.h>
#include <math.h>

int *a, *b;

typedef struct {
    int thread_id;
    int num_threads;
    int n;
    int start;
    int end;
} thread_arg_t;

void* vector_add(void* arg) {
    thread_arg_t* targ = (thread_arg_t*)arg;
    for(int i = targ->start; i < targ->end; i++) {
        a[i] += b[i];
    }
    return NULL;
}

double run_sequential(int n) {
    struct timeval t;
    double start, finish;
    
    gettimeofday(&t, NULL);
    start = t.tv_sec * 1000000.0 + t.tv_usec;
    
    for(int i = 0; i < n; i++) {
        a[i] += b[i];
    }
    
    gettimeofday(&t, NULL);
    finish = t.tv_sec * 1000000.0 + t.tv_usec;
    return (finish - start) / 1000.0; // ms
}

double run_parallel(int n, int num_threads) {
    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    thread_arg_t* args = malloc(num_threads * sizeof(thread_arg_t));
    
    int chunk_size = n / num_threads;
    int remainder = n % num_threads;
    int start = 0;
    
    for(int i = 0; i < num_threads; i++) {
        args[i].thread_id = i;
        args[i].num_threads = num_threads;
        args[i].n = n;
        args[i].start = start;
        
        int current_chunk = chunk_size + (i < remainder ? 1 : 0);
        args[i].end = start + current_chunk;
        start = args[i].end;
    }
    
    struct timeval t;
    double start_time, finish;
    
    gettimeofday(&t, NULL);
    start_time = t.tv_sec * 1000000.0 + t.tv_usec;
    
    for(int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, vector_add, &args[i]);
    }
    
    for(int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    gettimeofday(&t, NULL);
    finish = t.tv_sec * 1000000.0 + t.tv_usec;
    
    free(threads);
    free(args);
    
    return (finish - start_time) / 1000.0; // ms
}

int main() {
    int sizes[] = {1000, 10000, 100000, 1000000, 10000000, 100000000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int thread_counts[] = {1, 2, 4, 8, 16, 32, 64};
    int num_thread_counts = sizeof(thread_counts) / sizeof(thread_counts[0]);
    
    for(int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        
        a = (int*)malloc(n * sizeof(int));
        b = (int*)malloc(n * sizeof(int));
        
        for(int i = 0; i < n; i++) {
            a[i] = 2*i;
            b[i] = 2*i + 1;
        }
        
        printf("Размер вектора: %d\n", n);
        
        double seq_time = run_sequential(n);
        printf("  Последовательное время: %.3f мс\n", seq_time);
        
        double best_time = seq_time;
        int best_threads = 1;
        
        for(int t = 0; t < num_thread_counts; t++) {
            int num_threads = thread_counts[t];
            if(num_threads > n) continue;
            
            for(int i = 0; i < n; i++) {
                a[i] = 2*i;
            }
            
            double par_time = run_parallel(n, num_threads);
            double speedup = seq_time / par_time;
            
            printf("    %d потоков: %.3f мс, ускорение: %.2fx\n", 
                   num_threads, par_time, speedup);
            
            if(par_time < best_time) {
                best_time = par_time;
                best_threads = num_threads;
            }
        }
        
        if(seq_time < best_time * 1.1) { // 10% порог
            printf("  ВЫВОД: При размере %d распараллеливание неэффективно\n", n);
        } else {
            printf("  ВЫВОД: При размере %d оптимальное число потоков: %d\n", 
                   n, best_threads);
        }
        
        printf("\n");
        free(a);
        free(b);
    }
    
    return 0;
}