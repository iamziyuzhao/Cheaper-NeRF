import java.util.*;

public class InitGame {
    private int length;
    private char[][] cordnt;

    public InitGame(int length){
        this.length = length;
        this.cordnt = new char[length][length];
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



}
