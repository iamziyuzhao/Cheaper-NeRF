
package csc242_proj2;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class CNFreader {
    private List<List<Integer>> clauses;
    private List<Integer> currentClause;

    public CNFreader(String filename) throws IOException {
        clauses = new ArrayList<>();
        currentClause = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        String line;

        while ((line = reader.readLine()) != null) {
            line = line.trim();

            // Skip comments or p-lines
            if (line.startsWith("c") || line.startsWith("p") || line.startsWith("C")) {
                continue;
            }

            String[] literals = line.split("\\s+");
            if(!line.isEmpty()) {
                for (String literal : literals) {
                    int value = Integer.parseInt(literal);
                    if (value == 0) {
                        if (!currentClause.isEmpty()) {
                            clauses.add(new ArrayList<>(currentClause));
                            currentClause.clear();
                        }
                    } else {
                        currentClause.add(value);
                    }
                }
            }
        }
        if (!currentClause.isEmpty()) {
            clauses.add(currentClause);
        }

    }

    public List<List<Integer>> getClauses() {
        return clauses;
    }

}
