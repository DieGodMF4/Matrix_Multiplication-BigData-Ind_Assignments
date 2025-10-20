// Build (Linux/macOS):  cc -O2 -march=native -o mm_baseline mm_baseline.c
// Build (Windows MSVC): cl /O2 mm_baseline.c Psapi.lib
// Build (Windows MinGW): gcc -O2 -o mm_baseline.exe mm_baseline.c -lpsapi
// Run: ./mm_baseline 64 128 256 -r 3

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
  #include <windows.h>
  #include <psapi.h>
#else
  #include <sys/resource.h>
  #include <sys/time.h>
#endif

static const unsigned BASE_SEED = 403086;

static double* alloc_matrix(int n) {
    size_t bytes = (size_t)n * (size_t)n * sizeof(double);
    double* m = (double*)malloc(bytes);
    if (!m) {
        fprintf(stderr, "malloc failed (%zu bytes)\n", bytes);
        exit(1);
    }
    return m;
}

static void fill_matrix(double* M, int n, unsigned seed) {
    unsigned x = seed;
    size_t N = (size_t)n * (size_t)n;
    for (size_t i = 0; i < N; ++i) {
        x = 1664525u * x + 1013904223u;
        M[i] = (double)(x & 0x00FFFFFFu) / (double)0x01000000u;
    }
}

static void matmul_basic(const double* A, const double* B, double* C, int n) {
    for (int i = 0; i < n; ++i) {
        double* Ci = &C[(size_t)i*n];
        for (int j = 0; j < n; ++j) Ci[j] = 0.0;
        for (int k = 0; k < n; ++k) {
            double aik = A[(size_t)i*n + k];
            const double* Bk = &B[(size_t)k*n];
            for (int j = 0; j < n; ++j) {
                Ci[j] += aik * Bk[j];
            }
        }
    }
}

#ifndef _WIN32
static double ms_since(struct timespec a, struct timespec b) {
    return (b.tv_sec - a.tv_sec) * 1000.0 + (b.tv_nsec - a.tv_nsec) / 1e6;
}
#endif

#ifdef _WIN32
static void get_mem_mb(double* rss_now_mb, double* rss_peak_mb) {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        *rss_now_mb  = pmc.WorkingSetSize     / 1e6;
        *rss_peak_mb = pmc.PeakWorkingSetSize / 1e6;
    } else {
        *rss_now_mb = *rss_peak_mb = -1.0;
    }
}
#else
static void get_mem_mb(double* rss_now_mb, double* rss_peak_mb) {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
    #ifdef __APPLE__
        *rss_peak_mb = ru.ru_maxrss / (1024.0 * 1024.0); // bytes to MB
    #else
        *rss_peak_mb = ru.ru_maxrss / 1024.0;            // kB to MB
    #endif
        *rss_now_mb = -1.0; // not provided
    } else {
        *rss_now_mb = *rss_peak_mb = -1.0;
    }
}
#endif

int main(int argc, char** argv) {
    int repeats = 3;
    int sizes[256]; int nsizes = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-r") || !strcmp(argv[i], "--repeats")) {
            if (i + 1 < argc) repeats = atoi(argv[++i]);
        } else {
            sizes[nsizes++] = atoi(argv[i]);
        }
    }
    if (nsizes == 0) {
        fprintf(stderr, "Usage: %s <sizes...> [-r REPEATS]\n", argv[0]);
        return 1;
    }

    printf("lang,size,repeats,avg_time_ms,rss_now_mb,rss_peak_mb,allocated_mb\n");
        char filename[128];
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    strftime(filename, sizeof(filename), "results_c_%Y%m%d_%H%M%S.csv", t);

    FILE *fp = fopen(filename, "w");
    if (!fp) { perror("fopen"); return 1; }
    fprintf(fp, "lang,size,repeats,avg_time_ms,rss_now_mb,rss_peak_mb,allocated_mb\n");

    for (int si = 0; si < nsizes; ++si) {
        int n = sizes[si];
        double *A = alloc_matrix(n), *B = alloc_matrix(n), *C = alloc_matrix(n);
        size_t alloc_bytes = 3ull * (size_t)n * (size_t)n * sizeof(double);

        double sum_ms = 0.0;
        for (int r = 0; r < repeats; ++r) {
            fill_matrix(A, n, BASE_SEED + (unsigned)r);
            fill_matrix(B, n, BASE_SEED + 1u + (unsigned)r);

        #ifdef _WIN32
            LARGE_INTEGER freq, t0, t1;
            QueryPerformanceFrequency(&freq);
            QueryPerformanceCounter(&t0);
            matmul_basic(A, B, C, n);
            QueryPerformanceCounter(&t1);
            double elapsed_ms = (t1.QuadPart - t0.QuadPart) * 1000.0 / (double)freq.QuadPart;
            sum_ms += elapsed_ms;
        #else
            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            matmul_basic(A, B, C, n);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            sum_ms += ms_since(t0, t1);
        #endif
        }

        double rss_now_mb, rss_peak_mb;
        get_mem_mb(&rss_now_mb, &rss_peak_mb);

        printf("c,%d,%d,%.3f,%.2f,%.2f,%.2f\n",
               n, repeats, sum_ms / repeats,
               rss_now_mb, rss_peak_mb, alloc_bytes / (1024.0*1024.0));
        fprintf(fp, "c,%d,%d,%.3f,%.2f,%.2f,%.2f\n",
                n, repeats, sum_ms / repeats,
                rss_now_mb, rss_peak_mb, alloc_bytes / (1024.0*1024.0));

        free(A); free(B); free(C);
    }
    fclose(fp);
    printf("Results saved to %s\n", filename);

    return 0;
}
