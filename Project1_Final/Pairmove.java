public class Pairmove {
    private Pointen r_point;
    private Pointen t_point;

    public  Pairmove(){

    }

    public Pairmove(Pointen r_point,Pointen t_point){
        this.r_point = r_point;
        this.t_point = t_point;
    }
    public Pointen getR_point(){
        return r_point;
    }
    public Pointen getT_point(){
        return t_point;
    }
}
