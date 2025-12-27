package qrels;

public interface EvalMetricTieBreaker {
    String name();
    double[][] transform(double[] values);
}
