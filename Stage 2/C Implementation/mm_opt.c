// Build (Windows MinGW): gcc -O3 -march=native -o mm_opt.exe mm_opt.c -lpsapi
// Build (Linux/macOS):   cc -O3 -march=native -o mm_opt mm_opt.c
// Usage:
//   ./mm_opt basic 128 256 -r 3
//   ./mm_opt blocked 128 256 --block 64 -r 3
//   ./mm_opt transposed 128 256 -r 3
//   ./mm_opt sparse 1024 --density 0.05 -r 3

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

static double frand(unsigned *x){ *x=1664525u*(*x)+1013904223u; return (double)(*x&0x00FFFFFFu)/16777216.0; }

static double** alloc_dense(int n){
    double** M = (double**)malloc(n*sizeof(double*));
    double*  D = (double*)calloc((size_t)n*(size_t)n,sizeof(double));
    for(int i=0;i<n;i++) M[i]=D+(size_t)i*n;
    return M; // free(M[0]); free(M);
}
static void free_dense(double** M){ free(M[0]); free(M); }

static void fill_dense(double** M, int n, unsigned seed){
    unsigned x=seed;
    for(size_t i=0;i<(size_t)n*(size_t)n;i++) M[0][i]=frand(&x);
}

static void matmul_basic(double**A,double**B,double**C,int n){
    for(int i=0;i<n;i++){
        double* Ci=C[i]; double* Ai=A[i];
        for(int k=0;k<n;k++){
            double aik=Ai[k]; double* Bk=B[k];
            for(int j=0;j<n;j++) Ci[j]+=aik*Bk[j];
        }
    }
}

static void matmul_transposed(double**A,double**B,double**C,int n){
    double** BT=alloc_dense(n);
    for(int i=0;i<n;i++) for(int j=0;j<n;j++) BT[i][j]=B[j][i];
    for(int i=0;i<n;i++){
        double* Ai=A[i]; double* Ci=C[i];
        for(int j=0;j<n;j++){
            double s=0.0; double* btj=BT[j];
            for(int k=0;k<n;k++) s+=Ai[k]*btj[k];
            Ci[j]=s;
        }
    }
    free_dense(BT);
}

static void matmul_blocked(double**A,double**B,double**C,int n,int BS){
    for(int ii=0; ii<n; ii+=BS)
    for(int kk=0; kk<n; kk+=BS)
    for(int jj=0; jj<n; jj+=BS){
        int im = (ii+BS<n)? ii+BS : n;
        int km = (kk+BS<n)? kk+BS : n;
        int jm = (jj+BS<n)? jj+BS : n;
        for(int i=ii;i<im;i++){
            double* Ci=C[i]; double* Ai=A[i];
            for(int k=kk;k<km;k++){
                double aik=Ai[k]; double* Bk=B[k];
                for(int j=jj;j<jm;j++) Ci[j]+=aik*Bk[j];
            }
        }
    }
}

/* --- CSR for A --- */
typedef struct { int n; int *row_ptr; int *col_idx; double *val; } CSR;

static CSR gen_csr(int n, double density, unsigned seed){
    CSR S; S.n=n;
    int nnz_target = (int)((double)n*(double)n*density + 0.5);
    int per = (nnz_target>=0? nnz_target:0) / n;
    int left = nnz_target - per*n;
    S.row_ptr=(int*)malloc((n+1)*sizeof(int));
    // allocate upper bound and then shrink
    int cap = (nnz_target>1? nnz_target:1);
    S.col_idx=(int*)malloc(cap*sizeof(int));
    S.val=(double*)malloc(cap*sizeof(double));
    int nnz=0; unsigned x=seed;
    for(int i=0;i<n;i++){
        int k = per + (i<left?1:0);
        S.row_ptr[i]=nnz;
        if(k>n) k=n;
        // simple spaced columns (pseudo-random)
        for(int t=0;t<k;t++){
            int j = (int)(frand(&x)*n);
            // avoid duplicates by linear probe
            int attempts=0;
            while(attempts<k && j>=S.row_ptr[i] && j<nnz && S.col_idx[j]==j) { j=(j+1)%n; attempts++; }
            if(nnz==cap){ cap*=2; S.col_idx=(int*)realloc(S.col_idx,cap*sizeof(int)); S.val=(double*)realloc(S.val,cap*sizeof(double)); }
            S.col_idx[nnz]=j; S.val[nnz]=frand(&x); nnz++;
        }
    }
    S.row_ptr[n]=nnz;
    return S;
}

static void spmm_csr_dense(CSR A, double** B, double** C, int n){
    for(int i=0;i<n;i++){
        double* Ci=C[i];
        for(int p=A.row_ptr[i]; p<A.row_ptr[i+1]; p++){
            int k=A.col_idx[p]; double aik=A.val[p]; double* Bk=B[k];
            for(int j=0;j<n;j++) Ci[j]+=aik*Bk[j];
        }
    }
}

#ifdef _WIN32
static double now_ms(){ LARGE_INTEGER f,t; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t); return (double)t.QuadPart*1000.0/(double)f.QuadPart; }
static void mem_now_peak(double*now_mb,double*peak_mb){
    PROCESS_MEMORY_COUNTERS pmc;
    if(GetProcessMemoryInfo(GetCurrentProcess(),&pmc,sizeof(pmc))){ *now_mb=pmc.WorkingSetSize/1e6; *peak_mb=pmc.PeakWorkingSetSize/1e6; } else { *now_mb=*peak_mb=-1.0; }
}
#else
static double now_ms(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1000.0 + t.tv_nsec/1e6; }
static void mem_now_peak(double*now_mb,double*peak_mb){
    struct rusage ru; if(getrusage(RUSAGE_SELF,&ru)==0){
    #ifdef __APPLE__
        *peak_mb = ru.ru_maxrss/(1024.0*1024.0);
    #else
        *peak_mb = ru.ru_maxrss/1024.0;
    #endif
        *now_mb = -1.0;
    } else { *now_mb=*peak_mb=-1.0; }
}
#endif

int main(int argc, char** argv){
    if(argc<3){ fprintf(stderr,"Usage: %s <basic|blocked|transposed|sparse> sizes... [-r R] [--block BS] [--density D]\n",argv[0]); return 1; }
    char algo[16]; strncpy(algo, argv[1], 15); algo[15]='\0';
    int repeats=3, BS=64; double density=0.05;
    int sizes[256]; int ns=0;
    for(int i=2;i<argc;i++){
        if(!strcmp(argv[i],"-r")||!strcmp(argv[i],"--repeats")) repeats=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--block")) BS=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--density")) density=strtod(argv[++i],NULL);
        else sizes[ns++]=atoi(argv[i]);
    }
    char fname[128]; time_t tt=time(NULL); struct tm* tmv=localtime(&tt);
    strftime(fname,sizeof(fname),"results_c_opt_%Y%m%d_%H%M%S.csv",tmv);
    FILE* fp=fopen(fname,"w"); fprintf(fp,"lang,algo,size,repeats,avg_time_ms,rss_now_mb,rss_peak_mb,extra\n");

    for(int s=0;s<ns;s++){
        int n=sizes[s];
        double sum=0.0;
        for(int r=0;r<repeats;r++){
            double start=now_ms();
            if(!strcmp(algo,"sparse")){
                CSR A=gen_csr(n,density,BASE_SEED);
                double** B=alloc_dense(n); fill_dense(B,n,BASE_SEED+1);
                double** C=alloc_dense(n);
                spmm_csr_dense(A,B,C,n);
                free_dense(B); free_dense(C); free(A.row_ptr); free(A.col_idx); free(A.val);
            }else{
                double** A=alloc_dense(n), **B=alloc_dense(n), **C=alloc_dense(n);
                fill_dense(A,n,BASE_SEED); fill_dense(B,n,BASE_SEED+1);
                if(!strcmp(algo,"blocked")) matmul_blocked(A,B,C,n,BS);
                else if(!strcmp(algo,"transposed")) matmul_transposed(A,B,C,n);
                else matmul_basic(A,B,C,n);
                free_dense(A); free_dense(B); free_dense(C);
            }
            double end=now_ms(); sum+= (end-start);
        }
        double now_mb, peak_mb; mem_now_peak(&now_mb,&peak_mb);
        fprintf(fp,"c,%s,%d,%d,%.3f,%.2f,%.2f,\"{block:%d,density:%.3f}\"\n",algo,n,repeats,sum/repeats,now_mb,peak_mb,BS,density);
    }
    fclose(fp); printf("! Saved %s\n",fname);
    return 0;
}
