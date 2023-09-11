import java.util.*;

public class InitGame {
    private int length;
    private char[][] cordnt;

    public InitGame(int length){
        this.length = length;
        this.cordnt = new char[length][length];
        initPawn();
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
             System.out.print(i + 1 + " ");
             for(int j = 0; j < this.length; j++){
                 if(cordnt[i][j]!='\0'){
                     System.out.print(cordnt[i][j] + " ");
                 }else{
                     System.out.print("  ");
                 }
             }
             System.out.println(i + 1 + " " );
         }

         //Initial the column
         System.out.print(" ");
         for(int i = 0; i < this.length; i++){
             char c = (char)(i+97);
             System.out.print(" " + c);
         }
        }



        public void initPawn(){
            if (length == 4){
                cordnt[1][0]=cordnt[1][1]=cordnt[1][2]=cordnt[1][3]='X';
                cordnt[2][0]=cordnt[2][1]=cordnt[2][2]=cordnt[2][3]='O';
            }else if(length == 5){
                cordnt[1][0]=cordnt[1][1]=cordnt[1][2]=cordnt[1][3]=cordnt[1][4]='X';
                cordnt[3][0]=cordnt[3][1]=cordnt[3][2]=cordnt[3][3]=cordnt[3][4]='O';
            }else{
                cordnt[1][0]=cordnt[1][1]=cordnt[1][2]=cordnt[1][3]=cordnt[1][4]=cordnt[1][5]=cordnt[1][6]=cordnt[1][7]='X';
                cordnt[6][0]=cordnt[6][1]=cordnt[6][2]=cordnt[6][3]=cordnt[6][4]=cordnt[6][5]=cordnt[6][6]=cordnt[6][7]='O';
            }
        }

}
