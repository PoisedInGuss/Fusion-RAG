package qrels;

import correlation.CrossProduct;
import correlation.RankScore;
import org.apache.commons.lang3.ArrayUtils;

import java.util.*;
import java.util.stream.Collectors;

public class SampledPermAggrTieBreaker extends PermAggrTieBreaker {
    static int[] SEEDS = { 314159265, 1414213 };
    static Random random = new Random(SEEDS[0]);
    public SampledPermAggrTieBreaker(int maxPermutations, int numRankings) { super(maxPermutations, numRankings); }

    @Override
    // A simpler implementation of shuffling
    public List<List<RankScore>> generateTieResolutions(List<RankScore> group) {
        List<List<RankScore>> permutations = new ArrayList<>();
        double min = group.stream().map(x->x.getScore()).min(Double::compareTo).get();

        for (int i = 0; i < maxPermutations; i++) {
            List<RankScore> shuffled = new ArrayList<>(group);
            System.out.println(shuffled);
            Collections.shuffle(shuffled, random);
            System.out.println(shuffled);
            for (int j=0; j < shuffled.size(); j++) {
                shuffled.get(j).setScore(group.get(j).getScore());
            }
            System.out.println(shuffled);
            permutations.add(shuffled);
        }
        System.out.println(permutations);
        return permutations;
    }

    List<RankScore> sample(Map<Double, List<List<RankScore>>> groupedWithTieResolutions) {
        List<RankScore> rankScoreList = new ArrayList<>(); // this is a flattened list
        for (Map.Entry<Double, List<List<RankScore>>> e: groupedWithTieResolutions.entrySet()) {
            List<List<RankScore>> vlist = e.getValue();
            // choose a particular element from this list -- this is a particular permutation of the tied values
            rankScoreList.addAll(vlist.get(random.nextInt(vlist.size())));
        }
        return rankScoreList;
    }

    public double[][] transform(double[] values) {
        double[][] evalMeasureMatrix = new double[numRankings][values.length];
        Map<Double, List<RankScore>> grouped = groupTiedValues(values);
        Map<Double, List<List<RankScore>>> groupedWithTieResolutions = new HashMap<>();
        Map<Double, List<Integer>> indexMap = new HashMap<>();

        for (Map.Entry<Double, List<RankScore>> e : grouped.entrySet()) {
            List<List<RankScore>> tieResolutions = generateTieResolutions(e.getValue());
            groupedWithTieResolutions.put(e.getKey(), tieResolutions);
        }

        System.out.println(groupedWithTieResolutions);

        for (int i=0; i < numRankings; i++) {
            // at each generative step, just choose a random index from each tied group
            List<RankScore> sample = sample(groupedWithTieResolutions);
            evalMeasureMatrix[i] = constructValues(sample);
            System.out.println(ArrayUtils.toString(evalMeasureMatrix[i]));
        }

        return evalMeasureMatrix;
    }

    double[] constructValues(List<RankScore> sample) {
        double[] values = new double[sample.size()];
        for (RankScore rs: sample) {
            values[rs.getId()] = rs.getScore();
        }
        return values;
    }

    public static void main(String[] args) {
        double[] values = {4.2, 2.1, 4.2, 3.3, 2.1, 3.3, 4.2, 4.2, 1.0, 2.1};
        double[][] tieResolvedValues = new SampledPermAggrTieBreaker(5, 10).transform(values);
        for (int i=0; i < tieResolvedValues.length; i++) {
            ArrayUtils.toString(tieResolvedValues[i]);
            System.out.println();
        }
    }

    @Override
    public String name() {
        return "unused";
    }
}
