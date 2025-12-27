package retrieval;

import correlation.OverlapStats;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.LMDirichletSimilarity;
import org.apache.lucene.store.FSDirectory;

import java.io.*;
import java.util.*;

class QueryNN {
    MsMarcoQuery query;
    List<MsMarcoQuery> nns;

    QueryNN(String line) {
        String[] tokens = line.split("\t");
        query = new MsMarcoQuery(tokens[0], tokens[5].split("\\|")[0]);
        nns = new ArrayList<>();
        nns.add(new MsMarcoQuery(tokens[2], tokens[5].split("\\|")[1]));
    }

    void add(String line) {
        String[] tokens = line.split("\t");
        nns.add(new MsMarcoQuery(tokens[2], tokens[5].split("\\|")[1]));
    }

    String process(OneStepRetriever retriever) throws IOException {
        StringBuilder sb = new StringBuilder();

        TopDocs topDocsA = retriever.getSearcher().search(query.getQuery(), 20);
        int rank = 1;
        for (MsMarcoQuery nn: nns) {
            TopDocs topDocsB = retriever.getSearcher().search(nn.getQuery(), 20);
            nn.simWithOrig = (float)OverlapStats.computeRBO(topDocsA, topDocsB);

            sb.append(String.format("%s\tQ0\t%s\t%d\t%.4f\t%s\n",
                    query.qid, nn.qid, rank++, nn.simWithOrig, query.qText + "| " + nn.qText));
        }
        return sb.toString();
    }
}

public class QueryRBO_NN {

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Arguments required: <Input file for NN> <Output file for RBO reranked NN>");
            System.out.println("using default arguments: <msmarco.nnq.small.res> <msmarco.nns.rbo.res>");
            args = new String[2];
            args[0] = "msmarco.nnq.small.res";
            args[1] = "msmarco.nns.rbo.res";
        }

        OneStepRetriever oneStepRetriever = new OneStepRetriever();
        oneStepRetriever.reader = DirectoryReader.open(FSDirectory.open(new File(Constants.MSMARCO_INDEX).toPath()));
        oneStepRetriever.searcher = new IndexSearcher(oneStepRetriever.reader);
        oneStepRetriever.getSearcher().setSimilarity(new LMDirichletSimilarity());

        BufferedReader br = new BufferedReader(new FileReader("msmarco.nnq.small.res"));
        BufferedWriter bw = new BufferedWriter(new FileWriter("msmarco.nns.rbo.res"));
        String line;

        String prev_qid = null;
        QueryNN qNN = null;

        int count = 0;
        while ((line=br.readLine())!=null) {
            String this_qid = line.split("\t")[0];
            if (prev_qid == null) {
                qNN = new QueryNN(line);
            }
            else if (!prev_qid.equals(this_qid)) {
                String qNNInfo = qNN.process(oneStepRetriever);
                bw.write(qNNInfo);
                qNN = new QueryNN(line);

                // logging
                count++;
                if (count%10 == 0)
                    System.out.println(String.format("Computed RBO scores for %d queries...\r", count));
            }
            else {
                qNN.add(line);
            }

            prev_qid = this_qid;
        }
        // write out the last bunch
        String qNNInfo = qNN.process(oneStepRetriever);
        bw.write(qNNInfo);

        br.close();
        bw.close();
    }
}
