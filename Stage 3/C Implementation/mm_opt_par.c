// Parallel Matrix Multiplication – Assignment 3
// Build (Windows): gcc -O3 -march=native -fopenmp -o mm_opt_par.exe mm_opt_par.c -lpsapi

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#else
#include <sys/resource.h>
#include <sys/time.h>
#endif

#define SEED 403086

double now_ms() {
#ifdef _WIN32
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * 1000.0 / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec*1000.0 + ts.tv_nsec/1e6;
#endif
}

double** allocM(int n) {
    double** M = malloc(n*sizeof(double*));
    double* d = calloc((size_t)n*n, sizeof(double));
    for (int i = 0; i < n; i++) M[i] = d + i*n;
    return M;
}

void freeM(double** M){ free(M[0]); free(M); }

void fillM(double** M, int n) {
    srand(SEED);
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            M[i][j] = (double)rand()/RAND_MAX;
}

// ---------------- PARALLEL KERNEL ----------------
void matmul_parallel(double** A, double** B, double** C, int n) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        double* Ci = C[i];
        double* Ai = A[i];
        for (int k = 0; k < n; k++) {
            double aik = Ai[k];
            double* Bk = B[k];

#pragma omp simd
            for (int j = 0; j < n; j++)
                Ci[j] += aik * Bk[j];
        }
    }
}

void mem_peak(double* nowMB, double* peakMB){
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    *nowMB = pmc.WorkingSetSize / 1e6;
    *peakMB = pmc.PeakWorkingSetSize / 1e6;
#else
    struct rusage ru; 
    getrusage(RUSAGE_SELF, &ru);
    *peakMB = ru.ru_maxrss / 1024.0;
    *nowMB = -1;
#endif
}

int main(int argc, char** argv){
    if(argc < 4){
        printf("Usage: mm_opt_par.exe <size1 size2 ...> -p <threads> -r <repeats>\n");
        return 1;
    }

    int sizes[32], ns = 0;
    int p = 1, R = 3;

    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"-p")) p = atoi(argv[++i]);
        else if(!strcmp(argv[i],"-r")) R = atoi(argv[++i]);
        else sizes[ns++] = atoi(argv[i]);
    }

    char fname[128];
    sprintf(fname, "results_c_parallel_%dthreads.csv", p);
    FILE* fp = fopen(fname, "w");

    fprintf(fp,"lang,size,threads,repeats,avg_time_ms,speedup,efficiency,rss_peak_mb\n");

    double T1_cache[4096]={0};

    for(int s=0;s<ns;s++){
        int n = sizes[s];

        for(int r=0;r<R;r++){
            double** A = allocM(n);
            double** B = allocM(n);
            double** C = allocM(n);
            fillM(A,n); fillM(B,n);

            double t0 = now_ms();
            matmul_parallel(A,B,C,n);
            double t1 = now_ms();

            double dt = t1 - t0;

            if(p == 1) T1_cache[n] = dt;

            double speed = (p==1? 1.0 : T1_cache[n] / dt);
            double eff = speed / p;

            double nowMB, peakMB;
            mem_peak(&nowMB, &peakMB);

            fprintf(fp, "c,%d,%d,%d,%.3f,%.3f,%.3f,%.2f\n",
                n, p, R, dt, speed, eff, peakMB);

            freeM(A); freeM(B); freeM(C);
        }
    }

    fclose(fp);
    printf("Saved %s\n", fname);
    return 0;
}
