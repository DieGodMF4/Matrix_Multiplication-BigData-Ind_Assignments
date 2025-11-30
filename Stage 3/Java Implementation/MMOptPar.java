import java.io.FileWriter;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.Locale;

public class MMOptPar {

    static final int SEED = 403086;

    static double[][] alloc(int n){
        return new double[n][n];
    }

    static void fill(double[][] M, int n){
        java.util.Random r = new java.util.Random(SEED);
        for(int i=0;i<n;i++)
            for(int j=0;j<n;j++)
                M[i][j] = r.nextDouble();
    }

    static void parallelMul(double[][] A, double[][] B, double[][] C, int n, int threads)
            throws InterruptedException {

        ExecutorService pool = Executors.newFixedThreadPool(threads);
        AtomicInteger nextRow = new AtomicInteger(0);
        Semaphore sem = new Semaphore(threads);

        for (int t=0; t<threads; t++) {
            pool.submit(() -> {
                try {
                    sem.acquire();
                    int i;
                    while ((i = nextRow.getAndIncrement()) < n) {
                        double[] Ci = C[i];
                        double[] Ai = A[i];
                        for (int k=0; k<n; k++) {
                            double aik = Ai[k];
                            double[] Bk = B[k];
                            for (int j=0;j<n;j++)
                                Ci[j] += aik * Bk[j];
                        }
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    sem.release();
                }
            });
        }

        pool.shutdown();
        pool.awaitTermination(1, TimeUnit.HOURS);
    }

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.US);

        if(args.length < 4){
            System.out.println("Usage: java MMOptPar <sizes...> -p <threads> -r <repeats>");
            return;
        }

        int[] sizes = new int[32]; int ns=0;
        int p = 1, R = 3;

        for(int i=0;i<args.length;i++){
            if(args[i].equals("-p")) p = Integer.parseInt(args[++i]);
            else if(args[i].equals("-r")) R = Integer.parseInt(args[++i]);
            else sizes[ns++] = Integer.parseInt(args[i]);
        }

        String out = "results_java_parallel_"+p+"threads.csv";
        FileWriter fw = new FileWriter(out);
        fw.write("lang,size,threads,repeats,avg_time_ms,speedup,efficiency,memory_mb\n");

        java.util.Map<Integer,Double> T1 = new java.util.HashMap<>();

        for(int s=0;s<ns;s++){
            int n = sizes[s];

            for(int r=0;r<R;r++){
                double[][] A = alloc(n), B = alloc(n), C = alloc(n);
                fill(A,n); fill(B,n);

                long t0 = System.nanoTime();
                parallelMul(A,B,C,n,p);
                long t1 = System.nanoTime();

                double dt = (t1-t0)/1_000_000.0;

                if(p==1) T1.put(n, dt);

                Double base = T1.get(n);
                double speed = 1.0;
                if (p == 1) {
                    speed = 1.0;
                } else if (base != null) {
                    speed = base / dt;
                }
                double eff = speed / p;

                double mem = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1e6;

                fw.write(String.format("java,%d,%d,%d,%.3f,%.3f,%.3f,%.2f\n",
                        n,p,R,dt,speed,eff,mem));
            }
        }

        fw.close();
        System.out.println("Saved "+out);
    }
}
