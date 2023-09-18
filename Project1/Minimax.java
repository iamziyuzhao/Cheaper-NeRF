import java.util.*;

public class Minimax {
    private char color;
    private char  agent_color;
    private int strategy;
    private InitGame game;


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

        if(strategy == 1){
            System.out.println("Agent: I am doing the randomly choice...");
            pairmove =randomly(cordnt);

        }
        return pairmove;
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
