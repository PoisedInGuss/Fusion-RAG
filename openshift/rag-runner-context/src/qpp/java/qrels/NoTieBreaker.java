package qrels;

public class NoTieBreaker implements EvalMetricTieBreaker {

    @Override
    public String name() {
        return "tau";
    }

    @Override
    public double[][] transform(double[] values) {
        double[][] matrix = new double[1][values.length]; // Create a 1 x n matrix
        System.arraycopy(values, 0, matrix[0], 0, values.length); // Copy elements
        return matrix;
    }
}
