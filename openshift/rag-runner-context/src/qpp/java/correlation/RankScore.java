package correlation;

public class RankScore implements Comparable<RankScore> {
    int id;
    int rank;
    double score;

    public RankScore(int id, int rank, double score) { this.id = id; this.rank = rank; this.score = score; }

    public int getId() { return id; }
    public int getRank() { return rank; }

    @Override
    public int compareTo(RankScore o) {
        return Double.compare(this.score, o.score);
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("(").append(id).append("-> ").append(rank).append(", ").append(score).append(")");
        return sb.toString();
    }

    public void setScore(double score) { this.score = score; }
    public double getScore() { return score; }
}


