package qrels;

import correlation.CrossProduct;
import correlation.RankScore;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class PermAggrTieBreaker implements EvalMetricTieBreaker {
    int numRankings;
    int maxPermutations;
    double delta;

    public PermAggrTieBreaker(int maxPermutations, int numRankings) {
        this.maxPermutations = maxPermutations;
        this.numRankings = numRankings;
    }

    public Map<Double, List<RankScore>> groupTiedValues(double[] values) {
        Map<Double, List<RankScore>> groups = new LinkedHashMap<>();

        for (int i=0; i < values.length; i++) {
            double x = values[i];
            List<RankScore> rankScoreList = groups.get(x);
            if (rankScoreList == null) {
                rankScoreList = new ArrayList<>();
                groups.put(x, rankScoreList);
            }
            rankScoreList.add(new RankScore(i, 0, x));
        }

        List<Double> tiedValues = groups.keySet().stream().collect(Collectors.toList());
        delta = findMinDifference(tiedValues);
        delta /= 100;

        for (Map.Entry<Double, List<RankScore>> e: groups.entrySet()) {
            List<RankScore> rslist = e.getValue();
            double offset = 0;
            for (RankScore rs: rslist) {
                rs.setScore(rs.getScore() + offset);
                offset += delta;
            }
        }
        return groups;
    }

    public List<List<RankScore>> generateTieResolutions(List<RankScore> group) {
        List<List<RankScore>> permutations = new ArrayList<>();
        generatePermutationsWithMaxLimit(group, 0, permutations);
        return permutations;
    }

    private void generatePermutationsWithMaxLimit(List<RankScore> group, int start, List<List<RankScore>> result) {
        generatePermutations(group, start, result);
    }

    private void generatePermutations(List<RankScore> group, int start, List<List<RankScore>> result) {
        if (result.size()==maxPermutations)
            return;

        if (start == group.size() - 1) {
            result.add(new ArrayList<>(group));
            return;
        }
        for (int i = start; i < group.size(); i++) {
            Collections.swap(group, start, i);
            generatePermutations(group, start + 1, result);
            Collections.swap(group, start, i);
        }
    }

    public double findMinDifference(List<Double> values) {
        // Edge case: If the array has fewer than two elements, return 0
        if (values == null || values.size() < 2) {
            throw new IllegalArgumentException("Array must have at least two elements.");
        }

        double[] arr = values.stream().mapToDouble(Double::doubleValue).toArray();
        Arrays.sort(arr);
        double minDifference = Double.MAX_VALUE; // Initialize with a large value

        // Compare each adjacent pair in the sorted array
        for (int i = 1; i < arr.length; i++) {
            double diff = arr[i] - arr[i - 1];
            if (diff < minDifference) {
                minDifference = diff;
            }
        }

        return minDifference;
    }

    public static List<Integer> generateList(int max) {
        return IntStream.rangeClosed(0, max-1)
                .boxed()
                .collect(Collectors.toList());
    }

    public static void main(String[] args) {
        double[] values = {4.2, 2.1, 4.2, 3.3, 2.1, 3.3, 4.2, 4.2, 1.0, 2.1};
        double[][] tieResolvedValues = new PermAggrTieBreaker(5, 10).transform(values);
    }

    @Override
    public double[][] transform(double[] values) {
        Map<Double, List<RankScore>> grouped = groupTiedValues(values);
        Map<Double, List<List<RankScore>>> groupedWithTieResolutions = new HashMap<>();
        Map<Double, List<Integer>> indexMap = new HashMap<>();

        int i = 0, n;
        int numGroups = grouped.keySet().size();

        for (Map.Entry<Double, List<RankScore>> e: grouped.entrySet()) {
            n = 1;
            List<List<RankScore>> tieResolutions = generateTieResolutions(e.getValue());
            groupedWithTieResolutions.put(e.getKey(), tieResolutions);
            if (i==0 || i==numGroups-1 || i==numGroups/2)
                n = tieResolutions.size();
            indexMap.put(e.getKey(), generateList(n));
            i++;
        }

        System.out.println(groupedWithTieResolutions);
        List<List<Integer>> crossProduct = CrossProduct.computeCrossProduct(indexMap);

        // Print the result
        List<Double> keys = indexMap.keySet().stream().collect(Collectors.toList());
        int numRankings = crossProduct.size();
        int numValues = values.length;
        //System.out.println(numValues);
        double[][] evalMeasureMatrix = new double[numRankings][numValues];

        for (i=0; i < numRankings; i++) {
            List<Integer> tuple = crossProduct.get(i);
            int c = 0;
            for (int j = 0; j < tuple.size(); j++) {
                Double key = keys.get(j);
                List<Double> vlist = groupedWithTieResolutions.get(key).get(tuple.get(j)).stream().map(x->x.getScore()).collect(Collectors.toList());
                for (int k=0; k < vlist.size(); k++) {
                    evalMeasureMatrix[i][c++] = vlist.get(k).doubleValue();
                }
            }
        }


//        for (i=0; i < numRankings; i++) {
//            for (int j=0; j < numValues; j++) {
//                System.out.print(String.format("%.4f ", evalMeasureMatrix[i][j]));
//            }
//            System.out.println();
//        }

        return evalMeasureMatrix;
    }

    @Override
    public String name() {
        return "tau-aggr";
    }
}
