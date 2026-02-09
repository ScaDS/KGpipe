import de.uni_mannheim.minie.MinIE;
import de.uni_mannheim.minie.annotation.AnnotatedProposition;
import de.uni_mannheim.utils.coreNLP.CoreNLPUtils;

import edu.stanford.nlp.pipeline.StanfordCoreNLP;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.charset.StandardCharsets;
import java.util.List;

public class MinIEApp {

    public static void main(String[] args) {

        if (args.length != 2) {
            throw new IllegalArgumentException(
                "Usage: java MinIEApp <input_file> <output_file>"
            );
        }

        String inputFile = args[0];
        String outputFile = args[1];

        StanfordCoreNLP parser = CoreNLPUtils.StanfordDepNNParser();

        List<String> lines;
        try {
            lines = Files.readAllLines(Paths.get(inputFile), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new RuntimeException("Failed to read input file: " + inputFile, e);
        }

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile))) {

            for (String line : lines) {
                if (line.trim().isEmpty()) continue;

                MinIE minie = new MinIE(line, parser, MinIE.Mode.SAFE);

                writer.write("Input sentence: " + line);
                writer.newLine();
                writer.write("=============================");
                writer.newLine();
                writer.write("Extractions:");
                writer.newLine();

                for (AnnotatedProposition ap : minie.getPropositions()) {
                    writer.write("Triple: " + ap.getSubject() + " | " + ap.getRelation() + " | " + ap.getObject());
                    writer.newLine();
                    writer.write("Factuality: " + ap.getFactualityAsString());
                    writer.newLine();
                    writer.write("Attribution: " + (ap.getAttribution().getAttributionPhrase() != null
                            ? ap.getAttribution().toStringCompact() : "NONE"));
                    writer.newLine();
                    writer.write("-------------------------------");
                    writer.newLine();
                }

                writer.newLine();
            }

        } catch (IOException e) {
            throw new RuntimeException("Failed to write output file: " + outputFile, e);
        }

        System.out.println("DONE! Output written to " + outputFile);
    }
}
