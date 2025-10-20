
import java.io.FileWriter;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.Locale;
import java.util.Random;

/**
 * Basic O(n^3) matrix multiply benchmark (Java). Usage: java MMBaseline
 * <sizes...> [-r REPEATS] Examples: javac MMBaseline.java java MMBaseline 64
 * 128 256 -r 3
 */
public class MMBaseline {

    static final int BASE_SEED = 403086;

    static double[] allocMatrix(int n) {
        return new double[n * n];
    }

    static void fill(double[] M, int n, long seed) {
        Random rnd = new Random(seed);
        for (int i = 0; i < M.length; i++) {
            M[i] = rnd.nextDouble();
        }
    }

    static void matmulBasic(double[] A, double[] B, double[] C, int n) {
        Arrays.fill(C, 0.0);
        for (int i = 0; i < n; i++) {
            int ioff = i * n;
            for (int k = 0; k < n; k++) {
                double aik = A[ioff + k];
                int koff = k * n;
                for (int j = 0; j < n; j++) {
                    C[ioff + j] += aik * B[koff + j];
                }
            }
        }
    }

    static double usedHeapMB() {
        Runtime rt = Runtime.getRuntime();
        long used = rt.totalMemory() - rt.freeMemory();
        return used / 1_000_000.0; // MB (decimal)
    }

    public static void main(String[] args) {
        int repeats = 3;
        int[] sizes = new int[256];
        int nsizes = 0;

        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("-r") || args[i].equals("--repeats")) {
                if (i + 1 < args.length) {
                    repeats = Integer.parseInt(args[++i]);
                }
            } else {
                sizes[nsizes++] = Integer.parseInt(args[i]);
            }
        }
        if (nsizes == 0) {
            System.err.println("Usage: java MMBaseline <sizes...> [-r REPEATS]");
            System.exit(1);
        }

        System.out.println("lang,size,repeats,avg_time_ms,heap_now_mb,heap_peak_mb,allocated_mb");
        DateTimeFormatter fmt = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss");
        String outname = "results_java_" + fmt.format(LocalDateTime.now()) + ".csv";

        try (FileWriter fw = new FileWriter(outname)) {
            fw.write("lang,size,repeats,avg_time_ms,heap_now_mb,heap_peak_mb,allocated_mb\n");

            for (int si = 0; si < nsizes; si++) {
                int n = sizes[si];
                double[] A = allocMatrix(n);
                double[] B = allocMatrix(n);
                double[] C = allocMatrix(n);
                double allocMB = 3.0 * (double) n * (double) n * 8.0 / (1024.0 * 1024.0);

                double sumMs = 0.0;
                double heapPeak = usedHeapMB();

                for (int r = 0; r < repeats; r++) {
                    fill(A, n, BASE_SEED + r);
                    fill(B, n, BASE_SEED + 1L + r);
                    long t0 = System.nanoTime();
                    matmulBasic(A, B, C, n);
                    long t1 = System.nanoTime();
                    sumMs += (t1 - t0) / 1_000_000.0;
                    double now = usedHeapMB();
                    if (now > heapPeak) {
                        heapPeak = now;
                    }
                }
                double heapNow = usedHeapMB();
                fw.write(String.format(Locale.US,
                        "java,%d,%d,%.3f,%.2f,%.2f,%.2f%n",
                        n, repeats, sumMs / repeats, heapNow, heapPeak, allocMB));
            }
            System.out.println("Results saved to " + outname);
        } catch (IOException e) {
            System.err.println("I/O error writing results file: " + e.getMessage());
            System.exit(1);
        }
    }
}
