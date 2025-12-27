package correlation;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class SARE implements QPPCorrelationMetric {
    @Override
    public double correlation(double[] gt, double[] pred) {
        double sAre = computeSARE(gt, pred);
        return sAre;
    }

    @Override
    public String name() {
        return "SARE";
    }

    public double[] computeSAREPerQuery(double[] gt, double[] pred) {
        int n = gt.length;
        double[] rankDiffs = new double[n];
        RankScore[] gt_rs = new RankScore[n];
        RankScore[] pred_rs = new RankScore[n];

        for (int i=0; i < n; i++) {
            gt_rs[i] = new RankScore(i, i, gt[i]);
            pred_rs[i] = new RankScore(i, i, pred[i]);
        }

//        System.out.println();
//        Arrays.stream(gt_rs).forEach(System.out::print);
//        System.out.println();
//        Arrays.stream(pred_rs).forEach(System.out::print);
//        System.out.println();

        Arrays.sort(gt_rs);
        Arrays.sort(pred_rs);

        Map<Integer, RankScore> map_gts = new HashMap<>();
        Map<Integer, RankScore> map_preds = new HashMap<>();
        for (int i=0; i < n; i++) {
            gt_rs[i].rank = i;
            pred_rs[i].rank = i;
            map_gts.put(gt_rs[i].id, gt_rs[i]);
            map_preds.put(pred_rs[i].id, pred_rs[i]);
        }

//        System.out.println();
//        Arrays.stream(gt_rs).forEach(System.out::print);
//        System.out.println();
//        Arrays.stream(pred_rs).forEach(System.out::print);
//        System.out.println();

        for (Integer id: map_gts.keySet()) {
            int gt_rank = map_gts.get(id).rank;
            int pred_rank = map_preds.get(id).rank;
//            System.out.println(id + ": " + gt_rank + ", " + pred_rank);
            rankDiffs[id] = Math.abs(gt_rank - pred_rank)/(double)gt.length; // rank diff of ith query
        }
        return rankDiffs;
    }

    double computeSARE(double[] gt, double[] pred) { // error: lower the better
        double[] sarePerQuery = computeSAREPerQuery(gt, pred);
        return Arrays.stream(sarePerQuery).average().getAsDouble();
    }

    public static void main(String[] args) {
        double[] gt =   {0.32, 0.15, 0.67, 0.08, 0.96, 0.45};
        double[] pred = {0.22, 0.75, 0.47, 0.83, 0.16, 0.05};

        System.out.println(String.format("SARE: %.4f", (new SARE()).correlation(gt, pred)));
    }
}
