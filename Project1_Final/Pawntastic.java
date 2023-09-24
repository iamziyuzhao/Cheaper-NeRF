import java.util.*;

public class Pawntastic {
    public static void main(String[] args){
        //gameSize 4*4 5*5 8*8
        int length = 4;
        //chessPiece
        char userColor = ' ';
        int depth = 0;
        //--Ask game size from user
        Scanner sc = new Scanner(System.in);

        System.out.println("Welcome to Pawntasitic(by Ziyu Zhao and Zhen Zhang)");
        System.out.print("Which size you prefer?\n"
            + "1. 4*4\n"
            + "2. 5*5\n"
            + "3. 8*8\n");
        //todo: check the input from user
        Integer realSize = Integer.parseInt(sc.nextLine());
        // check whether the input size is valid
        while (realSize != 1 && realSize != 2 && realSize != 3)
        {
            System.out.println("Invalid input. Please try again.");
            System.out.print("Which size you prefer?\n"
            + "1. 4*4\n"
            + "2. 5*5\n"
            + "3. 8*8\n");
            realSize = Integer.parseInt(sc.nextLine());
        }
        System.out.println(realSize);
        if (realSize == 1){
            length = 4;
        }else if (realSize ==2){
            length = 5;
        }else if (realSize == 3){
            length = 8;
        }

        //--Ask which method from user
        System.out.println("Which method you prefer?\n"
            + "1. An agent that plays randomly\n"
            + "2. An agent that uses MINIMAX\n"
            + "3. An agent that uses H-MINIMAX with a fixed depth cutoff and alpha-beta pruning\n"
        );
        Integer realMethod = Integer.parseInt(sc.nextLine());
        // check whether the input method is valid
        while (realMethod != 1 && realMethod != 2 && realMethod != 3)
        {
            System.out.println("Invalid input. Please try again.");
            System.out.println("Which method you prefer?\n"
            + "1. An agent that plays randomly\n"
            + "2. An agent that uses MINIMAX\n"
            + "3. An agent that uses H-MINIMAX with a fixed depth cutoff and alpha-beta pruning\n"
            );
            realMethod = Integer.parseInt(sc.nextLine());
        }

        if (realMethod == 3)
        {
            System.out.println("In which depth do you want to cut (recommand 6 depths for 8x8 board in reasonable time): ");
            depth =(int) Integer.parseInt(sc.nextLine());
            while (depth < 0)
            {
                System.out.println("Invalid input. Please try again.");
                System.out.println("In which depth do you want to cut (recommand 6 depths for 8x8 board in reasonable time): ");
                depth =(int) Integer.parseInt(sc.nextLine());
            }
        }
            
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
        Minimax minimax = new Minimax(realMethod,userColor,length,depth);

        InitGame game = new InitGame(length,userColor,minimax);
        game.gameBoard();
        game.play_game();
        System.out.println(userColor);
    }
}
