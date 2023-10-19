import java.io.IOException;
import java.util.*;
public class main {
    public static void main(String[] args) throws IOException {
        String input;
        Scanner sc = new Scanner(System.in);
        System.out.println("----Part 1 CNF Tester----");
        System.out.println(
                "Here is the CNF file that professor gave for Part1: You can simply copy the path for input\n" +
                "./cnfs/cnf/aim-50-1_6-yes1-4.cnf\n" +
                "./cnfs/cnf/hole6.cnf\n" +
                "./cnfs/cnf/par8-1-c.cnf\n" +
                "./cnfs/cnf/quinn.cnf\n" +
                "./cnfs/cnf/zebra_v155_c1135.cnf");

        while (true) {
            System.out.println("\nPlease Enter the path of your CNF file(enter 'quit' to quit):");
            input = sc.nextLine();
            if("quit".equalsIgnoreCase(input)){
                System.out.println("");
                break;
            }
            CNFreader reader = new CNFreader(input);
            System.out.println("Clauses:");
            List<List<Integer>> kb = reader.getClauses();
            System.out.println(kb);
        }


        System.out.println("----Part 2 TT-ENTAILS----");
        // Q1
        System.out.println("Paths of CNF file for Q1 in Part 2:\n"+
                           "./cnfs/part2/pq.cnf");
        System.out.println("Q1. Please Enter the path of your CNF file");
        CNFreader reader2 = new CNFreader(sc.nextLine());
        List<List<Integer>> kb2 = reader2.getClauses();
        ModelChecker checker = new ModelChecker();
        List<List<Integer>> alpha = new ArrayList<>();
        alpha.add(Arrays.asList(2));
        System.out.println(checker.ttEntails(kb2, alpha) + "\n");
        alpha.clear();

        // Q2
        // P1,1 = 1     P1,2 = 2    P1,3 = 6
        // P2,1 = 3     P2,2 = 4
        // P3,1 = 5
        // B1,1 = 7     B1,2 = 9
        // B2,1 = 8
        System.out.println("Paths of CNF file for Q2 in Part 2:\n"+
                           "./cnfs/part2/ww.cnf");
        System.out.println("Q2. Please Enter the path of your CNF file");
        reader2 = new CNFreader(sc.nextLine());
        kb2 = reader2.getClauses();
        // The agent starts at [1,1]. Add perception: R4: ¬B1,1
        kb2.add(Arrays.asList(-7));
        // Entail ¬P1,2
        alpha.add(Arrays.asList(-2));
        System.out.println(checker.ttEntails(kb2, alpha) + "\n");
        alpha.clear();
        // Entail ¬P2,1
        alpha.add(Arrays.asList(-3));
        System.out.println(checker.ttEntails(kb2, alpha) + "\n");
        alpha.clear();
        // Not ential P2,2
        alpha.add(Arrays.asList(4));
        System.out.println(checker.ttEntails(kb2, alpha) + "\n");
        alpha.clear();
        // Not ential ¬P2,2
        alpha.add(Arrays.asList(-4));
        System.out.println(checker.ttEntails(kb2, alpha) + "\n");
        alpha.clear();

        // The agent moves to [2, 1]. Add perception: R5: B2,1
        kb2.add(Arrays.asList(8));
        // Ential P2,2 ∨ P3,1
        alpha.add(Arrays.asList(4,5));
        System.out.println(checker.ttEntails(kb2, alpha) + "\n");
        alpha.clear();
        // Not ential P2,2
        alpha.add(Arrays.asList(4));
        System.out.println(checker.ttEntails(kb2, alpha) + "\n");
        alpha.clear();
        // Not ential ¬P2,2
        alpha.add(Arrays.asList(-4));
        System.out.println(checker.ttEntails(kb2, alpha) + "\n");
        alpha.clear();
        // Not ential P3,1
        alpha.add(Arrays.asList(5));
        System.out.println(checker.ttEntails(kb2, alpha) + "\n");
        alpha.clear();
        // Not ential ¬P3,1
        alpha.add(Arrays.asList(-5));
        System.out.println(checker.ttEntails(kb2, alpha) + "\n");
        alpha.clear();

        // Move to [1, 2]. Add perception: R6: ¬B1,2
        kb2.add(Arrays.asList(-9));
        // Ential ¬P2,2
        alpha.add(Arrays.asList(-4));
        System.out.println(checker.ttEntails(kb2, alpha) + "\n");
        alpha.clear();
        // Ential P3,1
        alpha.add(Arrays.asList(5));
        System.out.println(checker.ttEntails(kb2, alpha) + "\n");
        alpha.clear();

        // Q3
        System.out.println("Paths of CNF file for Q3 in Part 2:\n"+
                           "./cnfs/part2/unicorn.cnf");
        System.out.println("Q3. Please Enter the path of your CNF file");
        reader2 = new CNFreader(sc.nextLine());
        kb2 = reader2.getClauses();
        alpha.add(Arrays.asList(1));
        System.out.println("The unicorn is mythical. " + checker.ttEntails(kb2, alpha) + "\n");
        alpha.clear();
        alpha.add(Arrays.asList(5));
        System.out.println("The unicorn is magical. " + checker.ttEntails(kb2, alpha) + "\n");
        alpha.clear();
        alpha.add(Arrays.asList(4));
        System.out.println("The unicorn is horned. " + checker.ttEntails(kb2, alpha));
        alpha.clear();


        System.out.println("----Part3 WalkSAT----");
        System.out.println("----Question 1 CNF for (x1 ∨ x3 ∨ ¬x4)∧(x4)∧(x2 ∨ ¬x3)----");
        while (true) {
            System.out.println("Enter 'quit' to quit, OR Enter anything to keep trying Question1:");
            input = sc.nextLine();

            System.out.println();
            if("quit".equalsIgnoreCase(input)){
                break;
            }else {
                input = "p3q1.cnf";
                CNFreader reader = new CNFreader(input);
                System.out.println("Clauses:");
                List<List<Integer>> kb = reader.getClauses();
                System.out.println(kb);
                SatTest sat = new SatTest();
                sat.WalkSAT(kb);
            }
        }

        System.out.println("----Question 2 N-Queens----");
        System.out.println(
                "Here is the CNF file that professor gave for Q2: You can simply copy the path for input\n" +
                "./cnfs/nqueens/nqueens_4.cnf\n" +
                "./cnfs/nqueens/nqueens_8.cnf\n" +
                "./cnfs/nqueens/nqueens_12.cnf\n" +
                "./cnfs/nqueens/nqueens_16.cnf" );

        while (true) {
            System.out.println("\nPlease Enter the path of your CNF file(enter 'quit' to quit):");
            input = sc.nextLine();

            System.out.println();
            if("quit".equalsIgnoreCase(input)){
                break;
            }else {
                CNFreader reader = new CNFreader(input);
                System.out.println("Clauses:");
                List<List<Integer>> kb = reader.getClauses();
                System.out.println(kb);
                SatTest sat = new SatTest();
                sat.WalkSAT(kb);
            }
        }

        System.out.println("----Question 3 Pigeonhole problems----");
        System.out.println(
                "Here is the CNF file that professor gave for Q3: You can simply copy the path for input\n" +
                "./cnfs/pigeonhole/pigeonhole_1_1.cnf\n" +
                "./cnfs/pigeonhole/pigeonhole_2_2.cnf\n" +
                "./cnfs/pigeonhole/pigeonhole_3_3.cnf\n" +
                "to\n" +
                "./cnfs/pigeonhole/pigeonhole_19_19.cnf\n" +
                "./cnfs/pigeonhole/pigeonhole_20_20.cnf");
        while (true) {
            System.out.println("\nPlease Enter the path of your CNF file(enter 'quit' to quit):");
            input = sc.nextLine();

            System.out.println();
            if("quit".equalsIgnoreCase(input)){
                break;
            }else {
                CNFreader reader = new CNFreader(input);
                System.out.println("Clauses:");
                List<List<Integer>> kb = reader.getClauses();
                System.out.println(kb);
                SatTest sat = new SatTest();
                sat.WalkSAT(kb);
            }
        }



    }
}