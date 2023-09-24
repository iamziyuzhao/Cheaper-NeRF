import java.util.*;

public class Minimax {
    private char color;
    private char agent_color;
    private int strategy;
    private InitGame game;
    private int depth;

    public static final char BLACKPAWN = '\u265F';
    public static final char WHITEPAWN = '\u2659';
    public static final char EMPTY = ' ';

    public Minimax(int strategy, char color, int length, int depth){
        this.strategy = strategy;
        this.color = color;
        this.game = new InitGame(length);
        this.depth = depth;
        if(this.color == '\u2659'){
            agent_color = '\u265F';
        }else
            agent_color = '\u2659';
    }

    public Pairmove makeDecision(char[][] cordnt){

        Pairmove pairmove = new Pairmove();

        System.out.println("");

        switch (strategy) {
            case 1:
                System.out.println("Agent: I am making the randomly choice...");
                pairmove = randomly(cordnt);
                break;
            
            case 2:
                System.out.println("Agent: I am making a choice with MINIMAX...");
                // cutoffDepth isn't used while using minimax, so it's default to 0
                pairmove = useMinimax(cordnt, this.agent_color, false, 0);
                break;

            case 3:
                System.out.println("Agent: I am making a choice with H-MINIMAX...");
                pairmove = useMinimax(cordnt, this.agent_color, true, this.depth);
                break;

            default:
                break;
        }
        
        return pairmove;
    }

    // Both minmax and hMinimax can use this function to run
    public Pairmove useMinimax(char[][] board, char whoseTurn, boolean useAlphaBetaPruning, int cutoffDepth) {
        Pairmove bestMove = new Pairmove();
        //int bestValue = (whoseTurn == WHITEPAWN) ? Integer.MIN_VALUE : Integer.MAX_VALUE;
        int bestValue = Integer.MIN_VALUE;

        Map<Pointen, ArrayList<Pointen>> validMoves = game.moveAble_list(board, whoseTurn);
        
        for (Pointen start : validMoves.keySet()) {
            for (Pointen end : validMoves.get(start)) {
                int endR = end.getRow();
                int endC = end.getColumn();
                int startR = start.getRow();
                int startC = start.getColumn();
                // Apply the movex
                char tmp = board[endR][endC];
                board[endR][endC] = whoseTurn;
                board[startR][startC] = EMPTY;
    
                int moveValue;
                if (useAlphaBetaPruning) {
                    moveValue = hMinimax(board, 1, cutoffDepth, Integer.MIN_VALUE, Integer.MAX_VALUE, true);
                } else {
                    moveValue = minimax(board, whoseTurn, true);
                }
    
                // Undo the move
                board[endR][endC] = tmp;
                board[startR][startC] = whoseTurn;
    
                if (moveValue > bestValue) {
                    bestValue = moveValue;
                    bestMove = new Pairmove(start, end);
                }
            }
        }
    
        return bestMove;
    }    

    public int hMinimax(char[][] board, int depth, int cutoffDepth, int alpha, int beta, boolean isMax) {
        int score = evaluate(board);
    
        // if terminate
        if (score == 10 || 
            score == -10 || 
            depth == cutoffDepth) 
            return score;
        
        // if no move is left
        if (game.moveAble_list(board, WHITEPAWN).isEmpty() && 
            game.moveAble_list(board, BLACKPAWN).isEmpty())
            return 0;
    
        if (isMax) {
            return evaluateMovesWithPruning(board, depth, cutoffDepth, alpha, beta, WHITEPAWN, true);
        } else {
            return evaluateMovesWithPruning(board, depth, cutoffDepth, alpha, beta, BLACKPAWN, false);
        }
    }
    
    private int evaluateMovesWithPruning(char[][] board, int depth, int cutoffDepth, int alpha, int beta, char pawn, boolean isMaximizing) {
        Map<Pointen, ArrayList<Pointen>> validMoves = game.moveAble_list(board, pawn);
        int bestEval = isMaximizing ? Integer.MIN_VALUE : Integer.MAX_VALUE;
        for (Pointen start : validMoves.keySet()) {
            for (Pointen end : validMoves.get(start)) {
                // Apply the move
                char tmp = board[end.getRow()][end.getColumn()];
                board[end.getRow()][end.getColumn()] = pawn;
                board[start.getRow()][start.getColumn()] = EMPTY;
    
                int eval = hMinimax(board, depth + 1, cutoffDepth, alpha, beta, !isMaximizing);
    
                // Undo the move
                board[end.getRow()][end.getColumn()] = tmp;
                board[start.getRow()][start.getColumn()] = pawn;
    
                if (isMaximizing) {
                    bestEval = Math.max(bestEval, eval);
                    alpha = Math.max(alpha, eval);
                } else {
                    bestEval = Math.min(bestEval, eval);
                    beta = Math.min(beta, eval);
                }
                //System.out.println("alpha: " + alpha + " ;beta: " + beta);
                if (beta <= alpha) break;  // Alpha-beta pruning
            }
        }
        //System.out.println("alpha: " + alpha + " ;beta: " + beta);
        return bestEval;
    }

    public boolean isTerminal(char[][] board)
    {
        int boardSize = board[0].length;
        for (int c = 0; c < boardSize; c++)
            if (board[0][c] != EMPTY || board[boardSize-1][c] != EMPTY)
                return true;

        return false;
    }

    private static final int WHITE_WIN = 10;
    private static final int BLACK_WIN = -10;
    private int minimax(char[][] board, char whoseTurn, boolean isMax) {
        int bestVal = isMax ? Integer.MIN_VALUE : Integer.MAX_VALUE; //best value
        
        int score = evaluate(board);
        if (score == WHITE_WIN || score == BLACK_WIN) return score;

        // If no move left
        if (game.moveAble_list(board, WHITEPAWN).isEmpty() && 
            game.moveAble_list(board, BLACKPAWN).isEmpty())
            return 0;

        return findMove(board, bestVal, whoseTurn, isMax);
    }
    
    public int findMove(char[][] board, int bestVal, char whoseTurn, boolean isMax)
    {
        Map<Pointen, ArrayList<Pointen>> validMoves = game.moveAble_list(board, whoseTurn);

        for (Pointen start : validMoves.keySet()) 
        {
            for (Pointen end : validMoves.get(start)) 
            {
                // Move
                char tmp = board[end.getRow()][end.getColumn()];
                board[end.getRow()][end.getColumn()] = whoseTurn;
                board[start.getRow()][start.getColumn()] = EMPTY;

                int currentVal = minimax(board, whoseTurn, !isMax);

                if (isMax)
                    if (currentVal > bestVal)
                        bestVal = currentVal;
                else
                    if (currentVal < bestVal)
                        bestVal = currentVal;
                
                // Undo
                board[end.getRow()][end.getColumn()] = tmp;
                board[start.getRow()][start.getColumn()] = whoseTurn;
            }
        }
        return bestVal;
    }

    private int evaluate(char[][] board) {
        for (int i = 0; i < board.length; i++) {
            if (board[0][i] == BLACKPAWN) return -10;
            if (board[board.length - 1][i] == WHITEPAWN) return 10;
        }
        return 0;
    }

    public Pairmove randomly(char[][] cordnt) {
        Map<Pointen,ArrayList<Pointen>> moveAble = game.moveAble_list(cordnt, agent_color);

        Pointen randomKey = getRandomKey(moveAble);
        Pointen randomValue = getRandomValue(moveAble, randomKey);
        Pairmove pairmove = new Pairmove(randomKey,randomValue);

        return pairmove;
    }

    public Pointen getRandomKey(Map<Pointen,ArrayList<Pointen>> resultMap){
        Random random = new Random();
        Object[] keys = resultMap.keySet().toArray();
        Pointen key = (Pointen) keys[random.nextInt(keys.length)];

        return key;
    }
    public Pointen getRandomValue(Map<Pointen,ArrayList<Pointen>> resultMap, Pointen randomKey){
        ArrayList<Pointen> values = resultMap.get(randomKey);
        Random random = new Random();
        Pointen value = values.get(random.nextInt(values.size()));

        return value;
    }


}
