import java.util.*;

public class Pawntastic {
    public static void main(String[] args){
        //gameSize 4*4 5*5 8*8
        int length;
        //chessPiece
        char userColor = ' ';

        //--Ask game size from user
        Scanner sc = new Scanner(System.in);

        System.out.println("Welcome to Pawntasitic(by Ziyu Zhao and Tan Zhen)");
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

        //--Ask which method from user
        System.out.println("Which method you prefer?\n"
            + "1. An agent that plays randomly\n"
            + "2. An agent that uses MINIMAX\n"
            + "3. An agent that uses H-MINIMAX with a fixed depth cutoff and alpha-beta pruning\n"
        );
        Integer realMethod = Integer.parseInt(sc.nextLine());

        //--Ask which color from user
        System.out.println("which color you want to play(white for w or black for b):(white go first)");
        String color = sc.nextLine();
        if (color.equals("w")){
           userColor = '\u2659';
        }else if(color.equals("b")){
           userColor = '\u265F';
        }else{
            System.out.println("you are doing the wrong input please restart the program!");
        }
        System.out.println(userColor);
        Minimax minimax = new Minimax(realMethod,userColor,length);


        InitGame game = new InitGame(length,userColor,minimax);
        game.gameBoard();
        game.play_game();
        System.out.println(userColor);








    }
}
