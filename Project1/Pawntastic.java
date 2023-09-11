import java.util.*;

public class Pawntastic {
    public static void main(String[] args){
        //gameSize 4*4 5*5 8*8
        Integer[] gameSize = {1, 2, 3};
        int length = 0;
        //chessPiece
        String[] chessPiece = {"O", "X"};
        //

        //--Ask game size from user
        Scanner sc = new Scanner(System.in);

        System.out.println("Welcome to Pawntasitic");
        System.out.print("Which size you prefer?\n"
            + "1. 4*4\n"
            + "2. 5*5\n"
            + "3. 8*8\n");
        //todo: check the input from user
        Integer realSize = Integer.parseInt(sc.nextLine());
        System.out.println(realSize);
        if (realSize == 1){
            length = 4;
        }else if (realSize ==2){
            length = 5;
        }else{
            length = 8;
        }

//        //--Ask which method from user
//        System.out.println("Which method you prefer?\n"
//            + "1. MINIMAX\n"
//            + "2. H-MINIMAX with α/β");
//        //todo: check the input from user
//        Integer realMethod = Integer.parseInt(sc.nextLine());
//        System.out.println(realMethod);

        InitGame game = new InitGame(length);
        game.gameBoard();








    }
}
