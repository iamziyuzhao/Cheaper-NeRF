import java.io.IOException;
import java.util.*;
public class main {
    public static void main(String[] args) throws IOException {
        String filepath;
        Scanner sc = new Scanner(System.in);

        System.out.println("Please Enter the path of your CNF file");
        CNFreader reader = new CNFreader(sc.nextLine());
        System.out.println("Clauses:");
        System.out.println(reader.getClauses());
    }

}
