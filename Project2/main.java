// package csc242_proj2;
import java.io.IOException;
import java.util.*;

public class main {
    public static void main(String[] args) throws IOException {
        String filepath;
        Scanner sc = new Scanner(System.in);

        System.out.println("Q1. Please Enter the path of your CNF file");
        CNFreader reader = new CNFreader(sc.nextLine());
        // System.out.println("Clauses:");
        List<List<Integer>> kb = reader.getClauses();
        // System.out.println(kb);
        // System.out.println();

        // Part 2
        ModelChecker checker = new ModelChecker();
        // Q1
        List<List<Integer>> alpha = new ArrayList<>();
        alpha.add(Arrays.asList(2));
        System.out.println(checker.ttEntails(kb, alpha) + "\n");
        alpha.clear();

        // Q2
        // P1,1 = 1     P1,2 = 2    P1,3 = 6
        // P2,1 = 3     P2,2 = 4
        // P3,1 = 5     
        // B1,1 = 7     B1,2 = 9
        // B2,1 = 8
        System.out.println("Q2. Please Enter the path of your CNF file");
        reader = new CNFreader(sc.nextLine());
        kb = reader.getClauses();
        // The agent starts at [1,1]. Add perception: R4: ¬B1,1
        kb.add(Arrays.asList(-7)); 
        // Entail ¬P1,2
        alpha.add(Arrays.asList(-2)); 
        System.out.println(checker.ttEntails(kb, alpha) + "\n");
        alpha.clear();
        // Entail ¬P2,1
        alpha.add(Arrays.asList(-3)); 
        System.out.println(checker.ttEntails(kb, alpha) + "\n");
        alpha.clear();
        // Not ential P2,2
        alpha.add(Arrays.asList(4)); 
        System.out.println(checker.ttEntails(kb, alpha) + "\n");
        alpha.clear();
        // Not ential ¬P2,2
        alpha.add(Arrays.asList(-4)); 
        System.out.println(checker.ttEntails(kb, alpha) + "\n");
        alpha.clear();
        
        // The agent moves to [2, 1]. Add perception: R5: B2,1
        kb.add(Arrays.asList(8)); 
        // Ential P2,2 ∨ P3,1
        alpha.add(Arrays.asList(4,5)); 
        System.out.println(checker.ttEntails(kb, alpha) + "\n");
        alpha.clear();
        // Not ential P2,2
        alpha.add(Arrays.asList(4)); 
        System.out.println(checker.ttEntails(kb, alpha) + "\n");
        alpha.clear();
        // Not ential ¬P2,2
        alpha.add(Arrays.asList(-4)); 
        System.out.println(checker.ttEntails(kb, alpha) + "\n");
        alpha.clear();
        // Not ential P3,1
        alpha.add(Arrays.asList(5)); 
        System.out.println(checker.ttEntails(kb, alpha) + "\n");
        alpha.clear();
        // Not ential ¬P3,1
        alpha.add(Arrays.asList(-5)); 
        System.out.println(checker.ttEntails(kb, alpha) + "\n");
        alpha.clear();

        // Move to [1, 2]. Add perception: R6: ¬B1,2
        kb.add(Arrays.asList(-9)); 
        // Ential ¬P2,2
        alpha.add(Arrays.asList(-4)); 
        System.out.println(checker.ttEntails(kb, alpha) + "\n");
        alpha.clear();
        // Ential P3,1
        alpha.add(Arrays.asList(5)); 
        System.out.println(checker.ttEntails(kb, alpha) + "\n");
        alpha.clear();

        // Q3
        System.out.println("Q3. Please Enter the path of your CNF file");
        reader = new CNFreader(sc.nextLine());
        kb = reader.getClauses();
        alpha.add(Arrays.asList(1));
        System.out.println("The unicorn is mythical. " + checker.ttEntails(kb, alpha) + "\n");
        alpha.clear();
        alpha.add(Arrays.asList(5));
        System.out.println("The unicorn is magical. " + checker.ttEntails(kb, alpha) + "\n");
        alpha.clear();
        alpha.add(Arrays.asList(4));
        System.out.println("The unicorn is horned. " + checker.ttEntails(kb, alpha));
        alpha.clear();

        // Part 3
        // SatTest sat = new SatTest();
	    // sat.WalkSAT(kb);
        
    }
}
