import java.awt.desktop.SystemEventListener;
import java.util.*;

public class InitGame {
    private int length;
    private char[][] cordnt;
    private char user_color;

    public InitGame(int length, char user_color){
        this.length = length;
        this.cordnt = new char[length][length];
        initPawn();
        this.user_color = user_color;

    }

     public void gameBoard(){
         System.out.print("\n");
         //Initial the row
         System.out.print(" ");
         for(int i = 0; i < this.length; i++){
             char c = (char)(i+97);
             System.out.print(" " + c);
         }
         System.out.println("");

         //Initial the column
         for(int i = 0; i < this.length; i++){
             System.out.print(" +");
             for(int s = 0; s < this.length; s++){
                 System.out.print("-+");
             }
             System.out.println("");
             System.out.print(i + 1 + "|");
             for(int j = 0; j < this.length; j++){
                 if(cordnt[i][j]!='\0'){
                     System.out.print(cordnt[i][j] + "|");
                 }else{
                     System.out.print(" |");
                 }
             }
             System.out.println(i + 1);
         }

         //Initial the row
         System.out.print(" ");
         for(int i = 0; i < this.length; i++){
             char c = (char)(i+97);
             System.out.print(" " + c);
         }
        }

        //Initial the Pawn at begining
        public void initPawn(){
            if (length == 4){
                //for black
                cordnt[1][0]=cordnt[1][1]=cordnt[1][2]=cordnt[1][3]='\u265F';
                //for white
                cordnt[2][0]=cordnt[2][1]=cordnt[2][2]=cordnt[2][3]='\u2659';
            }else if(length == 5){
                cordnt[1][0]=cordnt[1][1]=cordnt[1][2]=cordnt[1][3]=cordnt[1][4]='\u265F';
                cordnt[3][0]=cordnt[3][1]=cordnt[3][2]=cordnt[3][3]=cordnt[3][4]='\u2659';
            }else{
                cordnt[1][0]=cordnt[1][1]=cordnt[1][2]=cordnt[1][3]=cordnt[1][4]=cordnt[1][5]=cordnt[1][6]=cordnt[1][7]='\u265F';
                cordnt[6][0]=cordnt[6][1]=cordnt[6][2]=cordnt[6][3]=cordnt[6][4]=cordnt[6][5]=cordnt[6][6]=cordnt[6][7]='\u2659';
            }
        }

        //This function is for white
        public boolean checkVali(char[][] cordnt,Point r_point, Point t_point,char color){
            //Select pawn
            int r_row = r_point.getRow();
            int r_column = r_point.getColumn();
            //To where
            int t_row = t_point.getRow();
            int t_column = t_point.getColumn();


            //the white one
            if(color == '\u2659') {
                //check the color if its right color
                if (cordnt[r_row][r_column] == color) {
                    int differRow = t_row - r_row;
                    int differColumn = t_column - r_column;
                    //check if its legal move(do not care the specific move)

                    //the left most legal move
                    if (r_column == 0) {
                        if ((differRow == -1 && differColumn == 0) || (differRow == -2 && differColumn == 0) || (differRow == -1 && differColumn == 1)) {
                            //go diagonal one step and check the right diagonal black pawn
                            if (((differRow == -1 && differColumn == 1)) && (cordnt[t_row][t_column] == '\u265F')) {
                                return true;
                            }
                            //go straight one step,and check the front is empty
                            else if ((differRow == -1 && differColumn == 0) && (cordnt[t_row][t_column] == '\0')) {
                                return true;
                            }
                            //go straight two step, and check the front is empty and if it is first move
                            else if ((differRow == -2 && differColumn == 0) &&
                                    (cordnt[t_row + 1][t_column] == '\0') &&
                                    (cordnt[t_row][t_column] == '\0') &&
                                    (r_row == length - 2)) {
                                return true;
                            }
                            //else return false
                            else
                                return false;
                        } else {
                            return false;
                        }
                    }

                    //the right most legal move
                    else if (r_column == length) {
                        if ((differRow == -1 && differColumn == -1) || (differRow == -1 && differColumn == 0) || (differRow == -2 && differColumn == 0)) {
                            //go diagonal one step,and check the left diagonal black pawn
                            if ((differRow == -1 && differColumn == -1) && (cordnt[t_row][t_column] == '\u265F')) {
                                return true;
                            }
                            //go straight one step,and check the front is empty
                            else if ((differRow == -1 && differColumn == 0) && (cordnt[t_row][t_column] == '\0')) {
                                return true;
                            }
                            //go straight two step, and check the front is empty and if it is first move
                            else if ((differRow == -2 && differColumn == 0) &&
                                    (cordnt[t_row + 1][t_column] == '\0') &&
                                    (cordnt[t_row][t_column] == '\0') &&
                                    (r_row == length - 2)) {
                                return true;
                            }
                            //else return false
                            else
                                return false;

                        } else return false;
                    }

                    //the normal legal move
                    else {
                        //&& is logic and. || is logic or
                        if ((differRow == -1 && differColumn == -1) ||
                                (differRow == -1 && differColumn == 0) ||
                                (differRow == -2 && differColumn == 0) ||
                                (differRow == -1 && differColumn == 1)
                        ) {   //go diagonal one step,and check the diagonal black pawn
                            if (((differRow == -1 && differColumn == -1) || (differRow == -1 && differColumn == 1)) && (cordnt[t_row][t_column] == '\u265F')) {
                                return true;
                            }
                            //go straight one step,and check the front is empty
                            else if ((differRow == -1 && differColumn == 0) && (cordnt[t_row][t_column] == '\0')) {
                                return true;
                            }
                            //go straight two step, and check the front is empty and if it is first move
                            else if ((differRow == -2 && differColumn == 0) &&
                                    (cordnt[t_row + 1][t_column] == '\0') &&
                                    (cordnt[t_row][t_column] == '\0') &&
                                    (r_row == length - 2)) {
                                return true;
                            }
                            //else return false
                            else
                                return false;
                        } else return false;
                    }
                }
            }

            //the black one
            else if (color == '\u265F') {
                //check the color if its right color
                if(cordnt[r_row][r_column] == color){
                    int differRow = t_row - r_row;
                    int differColumn = t_column - r_column;
                    //check if its legal move(do not care the specific move)

                    //the left most legal move
                    if (r_column == 0) {
                        if ((differRow == 1 && differColumn == 0) || (differRow == 2 && differColumn == 0) || (differRow == 1 && differColumn == 1)) {
                            //go diagonal one step and check the right diagonal white pawn
                            if (((differRow == 1 && differColumn == 1)) && (cordnt[t_row][t_column] == '\u2659')) {
                                return true;
                            }
                            //go straight one step,and check the front is empty
                            else if ((differRow == 1 && differColumn == 0) && (cordnt[t_row][t_column] == '\0')) {
                                return true;
                            }
                            //go straight two step, and check the front is empty and if it is first move
                            else if ((differRow == 2 && differColumn == 0) &&
                                    (cordnt[t_row - 1][t_column] == '\0') &&
                                    (cordnt[t_row][t_column] == '\0') &&
                                    (r_row == 1)) {
                                return true;
                            }
                            //else return false
                            else
                                return false;
                        } else return false;

                    }

                    //the right most legal move
                    else if (r_column == length) {
                        if ((differRow == 1 && differColumn == -1) || (differRow == 1 && differColumn == 0) || (differRow == 2 && differColumn == 0)) {
                            //go diagonal one step,and check the left diagonal white pawn
                            if ((differRow == 1 && differColumn == -1) && (cordnt[t_row][t_column] == '\u2659')) {
                                return true;
                            }
                            //go straight one step,and check the front is empty
                            else if ((differRow == 1 && differColumn == 0) && (cordnt[t_row][t_column] == '\0')) {
                                return true;
                            }
                            //go straight two step, and check the front is empty and if it is first move
                            else if ((differRow == 2 && differColumn == 0) &&
                                    (cordnt[t_row - 1][t_column] == '\0') &&
                                    (cordnt[t_row][t_column] == '\0') &&
                                    (r_row == 1)) {
                                return true;
                            }
                            //else return false
                            else
                                return false;
                        } else return false;
                    }

                    //the normal legal move
                    else {
                        //&& is logic and. || is logic or
                        if ((differRow == 1 && differColumn == -1) ||
                                (differRow == 1 && differColumn == 0) ||
                                (differRow == 2 && differColumn == 0) ||
                                (differRow == 1 && differColumn == 1)
                        ) {   //go diagonal one step,and check the diagonal white pawn
                            if (((differRow == 1 && differColumn == -1) || (differRow == 1 && differColumn == 1)) && (cordnt[t_row][t_column] == '\u2659')) {
                                return true;
                            }
                            //go straight one step,and check the front is empty
                            else if ((differRow == 1 && differColumn == 0) && (cordnt[t_row][t_column] == '\0')) {
                                return true;
                            }
                            //go straight two step, and check the front is empty and if it is first move
                            else if ((differRow == 2 && differColumn == 0) &&
                                    (cordnt[t_row - 1][t_column] == '\0') &&
                                    (cordnt[t_row][t_column] == '\0') &&
                                    (r_row == 1)) {
                                return true;
                            }
                            //else return false
                            else
                                return false;
                        } else return false;
                    }

                }


            }


            return false;
        }



// black for '\u265F'
// First(white) for '\u2659'
        public void play_game(){
            Scanner sc = new Scanner(System.in);
            while (true){
                //white turn
                if (user_color == '\u2659'){
                    System.out.println("");
                    System.out.println("Next to play:  WHITE/\u2659");

                    //Measure the Elapsed time
                    System.out.print("your move: (if you want to move from b4 to b3(remember start point always in first), typing:b4b3)\n" + "(? for help)\n");
//                    long startTime = System.currentTimeMillis();
                    String move = sc.next();
                    System.out.println(move);
                    if(move.equals("?")){
                        System.out.println(
                                "There is 2 condition for the legal movement: \n"+
                                "1. only go forward,which means can't downward\n" +
                                "2. go diagonally only if there is a opponent pawn\n"
                        );
                    }else{
                        //Initial the movement
                        char[] movement = move.toCharArray();
                        int r_column = (int) (movement[0]) - 97;
                        int r_row = (int) (movement[1]) - 49;
                        int column = (int) (movement[2]) - 97;
                        int row = (int) (movement[3]) - 49;
                        Point r_point = new Point(r_row,r_column);
                        Point t_point = new Point(row,column);

                        //check the movement are legal
                        //NOTICE: row and column revers in 2-d array(cordnt)
                        if (checkVali(cordnt,r_point,t_point,user_color)) {
                            cordnt[r_row][r_column] = ' ';
                            cordnt[row][column] = '\u2659';
                            user_color = '\u265F';
                            gameBoard();
                        }else{
                            System.out.println("you do the wrong way!");
                        }
                    }
                }

                //black turn
                else if (user_color == '\u265F'){
                    System.out.println("");
                    System.out.println("Next to play:  BLACK/\u265F");

                    //Measure the Elapsed time
                    System.out.print("your move: (if you want to move from b4 to b3(remember start point always in first), typing:b4b3)\n" + "(? for help)\n");
//                    long startTime = System.currentTimeMillis();
                    String move = sc.next();
                    System.out.println(move);
                    if(move.equals("?")){
                        System.out.println(
                                "There is 2 condition for the legal movement: \n"+
                                        "1. only go forward,which means can't downward\n" +
                                        "2. go diagonally only if there is a opponent pawn\n"
                        );
                    }else{
                        //Initial the movement
                        char[] movement = move.toCharArray();
                        int r_column = (int) (movement[0]) - 97;
                        int r_row = (int) (movement[1]) - 49;
                        int column = (int) (movement[2]) - 97;
                        int row = (int) (movement[3]) - 49;
                        Point r_point = new Point(r_row,r_column);
                        Point t_point = new Point(row,column);

                        //check the movement are legal
                        //NOTICE: row and column revers in 2-d array(cordnt)
                        if (checkVali(cordnt,r_point,t_point,user_color)) {
                            cordnt[r_row][r_column] = ' ';
                            cordnt[row][column] = '\u265F';
                            user_color = '\u2659';
                            gameBoard();
                        }else{
                            System.out.println("you do the wrong way!");
                        }

                    }
                }


            }


        }




}
