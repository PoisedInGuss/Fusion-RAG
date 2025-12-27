package stochastic_qpp;

import correlation.KendalCorrelation;
import correlation.NDCGCorrelation;
import correlation.SARE;

import java.util.Arrays;

public class QPPMetricBundle {
    double tau;
    double ndcg;
    double[] perQuerySARE;

    public QPPMetricBundle(double tau, double ndcg, double[] perQuerySARE) {
        this.perQuerySARE = perQuerySARE;
        this.tau = tau;
        this.ndcg = ndcg;
    }

    public QPPMetricBundle(double[] gt, double[] pred) {
        tau = new KendalCorrelation().correlation(gt, pred);
        ndcg = new NDCGCorrelation().correlation(gt, pred);
        perQuerySARE = new SARE().computeSAREPerQuery(gt, pred);
    }

    public double tau() { return tau; }
    public double ndcg() { return ndcg; }
    public double[] getPerQuerySARE() { return perQuerySARE; }
    public double sare() { return Arrays.stream(perQuerySARE).average().getAsDouble(); }
    public double sarc() { return 1-Arrays.stream(perQuerySARE).average().getAsDouble(); } // 1-sare
}
