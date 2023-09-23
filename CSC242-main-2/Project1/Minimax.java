import java.util.*;

public class Minimax {
    private char color;
    private char  agent_color;
    private int strategy;
    private InitGame game;

    public static final char BLACKPAWN = '\u265F';
    public static final char WHITEPAWN = '\u2659';
    public static final char EMPTY = ' ';


    public Minimax(int strategy, char color, int length){
        this.strategy = strategy;
        this.color = color;
        this.game = new InitGame(length);
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
                pairmove = useMinimax(cordnt, this.agent_color, true, 6);
                break;

            default:
                break;
        }
        
        return pairmove;
    }

    // Both minmax and hMinimax can use this function to run
    public Pairmove useMinimax(char[][] board, char whoseTurn, boolean useAlphaBetaPruning, int cutoffDepth) {
        Pairmove bestMove = new Pairmove();
        int bestValue = (whoseTurn == WHITEPAWN) ? Integer.MIN_VALUE : Integer.MAX_VALUE;
    
        Map<Pointen, ArrayList<Pointen>> validMoves = game.moveAble_list(board, whoseTurn);
    
        for (Pointen start : validMoves.keySet()) {
            for (Pointen end : validMoves.get(start)) {
                // Apply the move
                char tmp = board[end.getRow()][end.getColumn()];
                board[end.getRow()][end.getColumn()] = whoseTurn;
                board[start.getRow()][start.getColumn()] = EMPTY;
    
                int moveValue;
                if (useAlphaBetaPruning) {
                    moveValue = hMinimax(board, 1, cutoffDepth, Integer.MIN_VALUE, Integer.MAX_VALUE, (whoseTurn == WHITEPAWN) ? BLACKPAWN : WHITEPAWN);
                } else {
                    moveValue = minimax(board, 1, (whoseTurn == WHITEPAWN) ? BLACKPAWN : WHITEPAWN);
                }
    
                // Undo the move
                board[end.getRow()][end.getColumn()] = tmp;
                board[start.getRow()][start.getColumn()] = whoseTurn;
    
                if ((whoseTurn == WHITEPAWN && moveValue > bestValue) || (whoseTurn == BLACKPAWN && moveValue < bestValue)) {
                    bestValue = moveValue;
                    bestMove = new Pairmove(start, end);
                }
            }
        }
    
        return bestMove;
    }    

    public int hMinimax(char[][] board, int depth, int cutoffDepth, int alpha, int beta, char whoseTurn) {
        int score = evaluate(board);
    
        // if terminate
        if (score == 10 || 
            score == -10 || 
            !isMovable(board) || 
            depth == cutoffDepth) 
            return score;
    
        if (whoseTurn == WHITEPAWN) {
            return evaluateMovesWithPruning(board, depth, cutoffDepth, alpha, beta, WHITEPAWN, true);
        } else {
            return evaluateMovesWithPruning(board, depth, cutoffDepth, alpha, beta, BLACKPAWN, false);
        }
    }
    
    private int evaluateMovesWithPruning(char[][] board, int depth, int cutoffDepth, int alpha, int beta, char pawn, boolean isMaximizing) {
        Map<Pointen, ArrayList<Pointen>> validMoves = game.moveAble_list(board, pawn);
        int bestEval = isMaximizing ? Integer.MIN_VALUE : Integer.MAX_VALUE;
        //System.out.print("1 alpha: " + alpha + " beta: " + beta);
        //System.out.println("depth: "+depth);
        for (Pointen start : validMoves.keySet()) {
            for (Pointen end : validMoves.get(start)) {
                // Apply the move
                char tmp = board[end.getRow()][end.getColumn()];
                board[end.getRow()][end.getColumn()] = pawn;
                board[start.getRow()][start.getColumn()] = EMPTY;
    
                int eval = hMinimax(board, depth + 1, cutoffDepth, alpha, beta, (pawn == WHITEPAWN) ? BLACKPAWN : WHITEPAWN);
    
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
                
                if (beta <= alpha) break;  // Alpha-beta pruning
            }
        }
        //System.out.println(" 2 alpha: " + alpha + " ;beta: " + beta);
        return bestEval;
    }
     
    public int minimax(char[][] board, int depth, char whoseTurn) {
        int score = evaluate(board);
        
        // Base Cases
        if (score == 10 || score == -10) return score;
        if (!isMovable(board)) return 0;
    
        // Recursive Cases
        if (whoseTurn == WHITEPAWN) {
            //return evaluatePawnMoves(board, depth, WHITEPAWN, Integer.MIN_VALUE, true);
            return evaluatePawnMoves(board, depth, WHITEPAWN, true);
        } else {
            //return evaluatePawnMoves(board, depth, BLACKPAWN, Integer.MAX_VALUE, false);
            return evaluatePawnMoves(board, depth, BLACKPAWN, false);
        }
    }
    
    private int evaluatePawnMoves(char[][] board, int depth, char pawn, boolean isMaximizing) {
        Map<Pointen, ArrayList<Pointen>> validMoves = game.moveAble_list(board, pawn);
        int bestEval = isMaximizing ? Integer.MIN_VALUE : Integer.MAX_VALUE;
        
        for (Pointen start : validMoves.keySet()) {
            for (Pointen end : validMoves.get(start)) {
                // Apply the move
                char tmp = board[end.getRow()][end.getColumn()];
                board[end.getRow()][end.getColumn()] = pawn;
                board[start.getRow()][start.getColumn()] = EMPTY;
    
                int eval = minimax(board, depth + 1, (pawn == WHITEPAWN) ? BLACKPAWN : WHITEPAWN);
    
                // Undo the move
                board[end.getRow()][end.getColumn()] = tmp;
                board[start.getRow()][start.getColumn()] = pawn;
    
                bestEval = isMaximizing ? Math.max(bestEval, eval) : Math.min(bestEval, eval);
            }
        }
        System.out.println("bestEval: " + bestEval);
        return bestEval;
    }
    
    private int evaluatePawnMoves(char[][] board, int depth, char pawn, int bestSoFar, boolean isMaximizing) {
        int boardSize = board.length;
        System.out.println("depth: " + depth);
        System.out.print("1 bestsoFar: " + bestSoFar);
        for (int i = 0; i < boardSize; i++) {
            for (int j = 0; j < boardSize; j++) {
                if (board[i][j] == pawn) {
                    // Try to move forward
                    int forward = (pawn == WHITEPAWN) ? 1 : -1;
                    if (isValidMove(board, i + forward, j)) {
                        //System.out.println("Try to move forward");
                        applyMove(board, i, j, i + forward, j, pawn);
                        int current = minimax(board, depth + 1, (pawn == WHITEPAWN) ? BLACKPAWN : WHITEPAWN);
                        undoMove(board, i, j, i + forward, j, pawn);
                        bestSoFar = isMaximizing ? Math.max(bestSoFar, current) : Math.min(bestSoFar, current);
                    }
    
                    // Try initial double move
                    if ((pawn == WHITEPAWN && i == 1) || (pawn == BLACKPAWN && i == boardSize - 2)) {
                        if (isValidMove(board, i + 2 * forward, j)) {
                            System.out.println("Try initial double move");
                            applyMove(board, i, j, i + 2 * forward, j, pawn);
                            int current = minimax(board, depth + 1, (pawn == WHITEPAWN) ? BLACKPAWN : WHITEPAWN);
                            undoMove(board, i, j, i + 2 * forward, j, pawn);
                            bestSoFar = isMaximizing ? Math.max(bestSoFar, current) : Math.min(bestSoFar, current);
                        }
                    }
    
                    // Try diagonal captures
                    for (int d = -1; d <= 1; d += 2) { // -1 for left, 1 for right
                        if (isValidCapture(board, i + forward, j + d, pawn)) {
                            System.out.println("Try diagonal captures");
                            applyMove(board, i, j, i + forward, j + d, pawn);
                            int current = minimax(board, depth + 1, (pawn == WHITEPAWN) ? BLACKPAWN : WHITEPAWN);
                            undoMove(board, i, j, i + forward, j + d, pawn);
                            bestSoFar = isMaximizing ? Math.max(bestSoFar, current) : Math.min(bestSoFar, current);
                        }
                    }
                }
            }
        }
        System.out.println("2 bestsoFar: " + bestSoFar);
        return bestSoFar;
    }
    
    private boolean isValidMove(char[][] board, int x, int y) {
        return x >= 0 && x < board.length && y >= 0 && y < board.length && board[x][y] == EMPTY;
    }
    
    private boolean isValidCapture(char[][] board, int x, int y, char pawn) {
        return x >= 0 && x < board.length && y >= 0 && y < board.length && 
               (pawn == WHITEPAWN ? board[x][y] == BLACKPAWN : board[x][y] == WHITEPAWN);
    }
    
    private void applyMove(char[][] board, int startX, int startY, int endX, int endY, char pawn) {
        board[startX][startY] = EMPTY;
        board[endX][endY] = pawn;
    }
    
    private void undoMove(char[][] board, int startX, int startY, int endX, int endY, char pawn) {
        board[endX][endY] = EMPTY;
        board[startX][startY] = pawn;
    }
    
    private int evaluate(char[][] board) {
        for (int i = 0; i < board.length; i++) {
            if (board[0][i] == BLACKPAWN) return -10;
            if (board[board.length - 1][i] == WHITEPAWN) return 10;
        }
        return 0;
    }

    private boolean isMovable(char[][] board) {
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board.length; j++)
                if (board[i][j] == EMPTY) 
                    return true;
        }
        return false;
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
