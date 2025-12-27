package retrieval;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

public class MsMarcoNNQueriesWriter {
    public static void main(String[] args) throws Exception {
        KNNRelModel knnRelModel = new KNNRelModel(); // load query index

        BufferedReader br = new BufferedReader(new FileReader(Constants.QUERIES_MSMARCO_TRAIN));
        BufferedWriter bw = new BufferedWriter(new FileWriter("msmarco.nnq.res"));
        String line;

        int numQueries = 0;
        while ((line=br.readLine())!=null) {
            String[] tokens = line.split("\t");
            MsMarcoQuery q = new MsMarcoQuery(tokens[0], tokens[1]);
            knnRelModel.findKNNOfQueriesAndComputeRBO(q, bw);

            numQueries++;
            if (numQueries%5000 == 0)
                System.out.println(String.format("Saved NN records for %d queries\r", numQueries));
        }

        br.close();
        bw.close();
    }
}
