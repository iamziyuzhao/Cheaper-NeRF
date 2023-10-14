package csc242_proj2;
import java.io.IOException;
import java.util.*;

public class main {
    public static void main(String[] args) throws IOException {
        String filepath;
        Scanner sc = new Scanner(System.in);

        System.out.println("Please Enter the path of your CNF file");
        CNFreader reader = new CNFreader(sc.nextLine());
        System.out.println("Clauses:");
        List<List<Integer>> kb = reader.getClauses();
        System.out.println(kb);

        // Part 2
        ModelChecker checker = new ModelChecker();
        List<List<Integer>> alpha = new ArrayList<>();
        List<Integer> element = new ArrayList<Integer>();
        element.add(2);
        alpha.add(element);
        System.out.println(checker.ttEntails(kb, alpha));
        
        // Part 3
        SatTest sat = new SatTest();
		sat.WalkSAT(kb);
        
    }
}
