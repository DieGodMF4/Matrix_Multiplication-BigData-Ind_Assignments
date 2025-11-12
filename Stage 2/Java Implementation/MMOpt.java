// javac MMOpt.java
// java MMOpt blocked 128 256 --block 64 -r 3
// java MMOpt transposed 128 256 -r 3
// java MMOpt sparse 1024 --density 0.05 -r 3

import java.io.FileWriter;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Locale;
import java.util.Random;

public class MMOpt {

    static final int BASE_SEED = 403086;

    static double[][] allocDense(int n) {
        double[][] M = new double[n][n];
        return M;
    }

    static void fillDense(double[][] M, int n, long seed) {
        Random rnd = new Random(seed);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                M[i][j] = rnd.nextDouble();
            }
        }
    }

    static void basic(double[][] A, double[][] B, double[][] C, int n) {
        for (int i = 0; i < n; i++) {
            double[] Ci = C[i], Ai = A[i];
            for (int k = 0; k < n; k++) {
                double aik = Ai[k];
                double[] Bk = B[k];
                for (int j = 0; j < n; j++) {
                    Ci[j] += aik * Bk[j];
                }
            }
        }
    }

    static void transposed(double[][] A, double[][] B, double[][] C, int n) {
        double[][] BT = allocDense(n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                BT[i][j] = B[j][i];
            }
        }
        for (int i = 0; i < n; i++) {
            double[] Ai = A[i], Ci = C[i];
            for (int j = 0; j < n; j++) {
                double s = 0.0;
                double[] btj = BT[j];
                for (int k = 0; k < n; k++) {
                    s += Ai[k] * btj[k];
                }
                Ci[j] = s;
            }
        }
    }

    static void blocked(double[][] A, double[][] B, double[][] C, int n, int BS) {
        for (int ii = 0; ii < n; ii += BS) {
            for (int kk = 0; kk < n; kk += BS) {
                for (int jj = 0; jj < n; jj += BS) {
                    int im = Math.min(ii + BS, n), km = Math.min(kk + BS, n), jm = Math.min(jj + BS, n);
                    for (int i = ii; i < im; i++) {
                        double[] Ci = C[i], Ai = A[i];
                        for (int k = kk; k < km; k++) {
                            double aik = Ai[k];
                            double[] Bk = B[k];
                            for (int j = jj; j < jm; j++) {
                                Ci[j] += aik * Bk[j];
                            }
                        }
                    }
                }
            }
        }
    }

    /* ---- CSR for A ---- */
    static class CSR {

        int n;
        int[] rowPtr, colIdx;
        double[] val;
    }

    static CSR genCSR(int n, double density, long seed) {
        Random rnd = new Random(seed);
        CSR S = new CSR();
        S.n = n;
        int nnz = (int) Math.max(1, Math.round(n * n * density));
        int per = nnz / n, left = nnz - per * n;
        S.rowPtr = new int[n + 1];
        S.colIdx = new int[nnz];
        S.val = new double[nnz];
        int p = 0;
        for (int i = 0; i < n; i++) {
            S.rowPtr[i] = p;
            int k = per + (i < left ? 1 : 0);
            for (int t = 0; t < k; t++) {
                int j = rnd.nextInt(n);
                S.colIdx[p] = j;
                S.val[p] = rnd.nextDouble();
                p++;
            }
        }
        S.rowPtr[n] = p;
        return S;
    }

    static void spmm(CSR A, double[][] B, double[][] C, int n) {
        for (int i = 0; i < n; i++) {
            double[] Ci = C[i];
            for (int p = A.rowPtr[i]; p < A.rowPtr[i + 1]; p++) {
                int k = A.colIdx[p];
                double aik = A.val[p];
                double[] Bk = B[k];
                for (int j = 0; j < n; j++) {
                    Ci[j] += aik * Bk[j];
                }
            }
        }
    }

    static long nowNs() {
        return System.nanoTime();
    }

    static double usedHeapMB() {
        Runtime rt = Runtime.getRuntime();
        return (rt.totalMemory() - rt.freeMemory()) / 1_000_000.0;
    }

    public static void main(String[] args) throws IOException {
        Locale.setDefault(Locale.US);
        if (args.length < 2) {
            System.err.println("Usage: java MMOpt <basic|blocked|transposed|sparse> sizes... [-r R] [--block BS] [--density D]");
            System.exit(1);
        }
        String algo = args[0];
        int repeats = 3, BS = 64;
        double density = 0.05;
        int[] sizes = new int[256];
        int ns = 0;
        for (int i = 1; i < args.length; i++) {
            if (args[i].equals("-r") || args[i].equals("--repeats")) {
                repeats = Integer.parseInt(args[++i]);
            } else if (args[i].equals("--block")) {
                BS = Integer.parseInt(args[++i]);
            } else if (args[i].equals("--density")) {
                density = Double.parseDouble(args[++i]);
            } else {
                sizes[ns++] = Integer.parseInt(args[i]);
            }
        }
        String ts = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String out = "results_java_opt_" + algo + "_" + ts + ".csv";
        try (FileWriter fw = new FileWriter(out)) {
            fw.write("lang,algo,size,repeats,avg_time_ms,heap_now_mb,extra\n");
            for (int s = 0; s < ns; s++) {
                int n = sizes[s];
                double sumMs = 0.0;
                for (int r = 0; r < repeats; r++) {
                    long t0 = nowNs();
                    if (algo.equals("sparse")) {
                        CSR A = genCSR(n, density, BASE_SEED);
                        double[][] B = allocDense(n);
                        fillDense(B, n, BASE_SEED + 1);
                        double[][] C = allocDense(n);
                        spmm(A, B, C, n);
                    } else {
                        double[][] A = allocDense(n), B = allocDense(n), C = allocDense(n);
                        fillDense(A, n, BASE_SEED);
                        fillDense(B, n, BASE_SEED + 1);
                        if (algo.equals("blocked")) {
                            blocked(A, B, C, n, BS);
                        } else if (algo.equals("transposed")) {
                            transposed(A, B, C, n);
                        } else {
                            basic(A, B, C, n);
                        }
                    }
                    long t1 = nowNs();
                    sumMs += (t1 - t0) / 1_000_000.0;
                }
                double heap = usedHeapMB();
                fw.write(String.format("java,%s,%d,%d,%.3f,%.2f,\"{block:%d,density:%.3f}\"\n", algo, n, repeats, sumMs / repeats, heap, BS, density));
            }
        }
        System.out.println("! Saved " + out);
    }
}
