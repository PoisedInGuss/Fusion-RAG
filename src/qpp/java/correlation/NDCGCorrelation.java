package correlation;

import java.util.Arrays;
import java.util.Comparator;
import java.util.function.Function;

public class NDCGCorrelation implements QPPCorrelationMetric {
    static Function<Double, Double> expScaling = rel -> Math.pow(2, rel) - 1;
    static Function<Double, Double> linearScaling = rel -> rel;

    public static double computeNDCG(double[] labels, double[] scores, Function<Double, Double> scalingFunction) {
        if (labels.length != scores.length || labels.length == 0) {
            throw new IllegalArgumentException("Labels and scores must have the same length and be non-empty.");
        }

        int n = labels.length;
        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; i++) {
            indices[i] = i;
        }

        // Sort indices based on scores in descending order
        Arrays.sort(indices, Comparator.comparingDouble(i -> -scores[i]));

        double dcg = computeDCG(labels, indices, scalingFunction);

        // Sort indices based on actual labels in descending order for ideal DCG
        Arrays.sort(indices, Comparator.comparingDouble(i -> -labels[i]));

        double idcg = computeDCG(labels, indices, scalingFunction);

        return idcg == 0 ? 0 : dcg / idcg;
    }

    private static double computeDCG(double[] labels, Integer[] indices, Function<Double, Double> scalingFunction) {
        double dcg = 0.0;
        for (int i = 0; i < indices.length; i++) {
            int idx = indices[i];
            dcg += scalingFunction.apply(labels[idx]) / (Math.log(i + 2) / Math.log(2));
        }
        return dcg;
    }

    public static void main(String[] args) {
        double[] labels = {2, 3, 0, 1, 2, 3};
        double[] scores = {0.8, 0.7, 0.6, 0.5, 0.4, 0.9};
        double ndcg = computeNDCG(labels, scores, expScaling);
    }

    @Override
    public double correlation(double[] gt, double[] pred) {
        return computeNDCG(gt, pred, expScaling);
    }

    @Override
    public String name() {
        return "ndcg";
    }
}
