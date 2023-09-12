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
        public boolean checkVali(char[][] cordnt,Point point,char color){
            return true;
        }

        public ArrayList<Point> moveable_list(char[][] cordnt, char color){
            ArrayList<Point> move_list = new ArrayList<Point>();
            for (int i = 0; i < this.length; i++) {
                for (int j = 0; j < this.length; j++) {
                    Point point = new Point(i, j);
                    if (checkVali(cordnt, point, color))
                        move_list.add(point);
                }
            }
            return move_list;

        }
        public char[][] actualMove(char[][] cordnt, Point point, char user_color){
            int column = point.getColumn();
            int row = point.getRow();
            char turnedColor = ' ';
            //Assign color
            cordnt[row][column] = user_color;
            //turn to opposite color
            if(user_color == '\u265F'){
                turnedColor = '\u2659';
            }else{
                turnedColor = '\u265F';
            }



            return null;
        }



// black for '\u265F'
// First(white) for '\u2659'
        public void play_game(){
            Scanner sc = new Scanner(System.in);
            while (true){
                //white turn
                if (user_color == '\u2659'){
                    ArrayList<Point> moveAble = moveable_list(cordnt,user_color);
                    //todo: check the moveAble list and resultr

                    //Notice the user the color
                    System.out.println(' ');
                    if (user_color == '\u2659') {
                        System.out.println("Next to play:  WHITE/\u2659");
                    }else{
                        System.out.println("Next to play:  Black/\u265F");
                    }

                    //Measure the Elapsed time
                    System.out.println("your move: ");
                    long startTime = System.currentTimeMillis();
                    String move = sc.next();
                    long endTime = System.currentTimeMillis();
                    long elapsedTime = endTime - startTime;
                    System.out.println("Elapsed time: " + elapsedTime+"Millisecond");

                    //todo: User Ask for help

                    //Initial the movement
                    char[] movement = move.toCharArray();
                    int column = (int)(movement[0])-97;
                    int row = (int)(movement[1])-49;
                    Point point = new Point(row,column);
                    cordnt = actualMove(cordnt,point,user_color);


                    //todo check badmovement and vali
                    gameBoard();







                }
            }


        }




}
