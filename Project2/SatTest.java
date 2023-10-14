package csc242_proj2;
import java.util.*;

public class SatTest {
	
	public void WalkSAT(List<List<Integer>> constraints){
		double p=GetP();
		int max_tries = GetTries();
		int max_flips = GetFlips();
		List<Integer> container = null;
		while((container==null)&&(max_tries>0)){
			container = WalkSAT_I(constraints, p, max_flips);
			max_tries--;
		}
		PrintBool(container);
	}
	private List<Integer> WalkSAT_I(List<List<Integer>> constraints, double p, int max_flips) {
		
		ModelChecker mc = new ModelChecker();
		
		//to get random assigned model
		Set<Integer> symbols = mc.getSymbols(constraints);
		List<Integer> model = randomModel(symbols);
		
		//to find the satisfied model
		for(int i=0; i<max_flips; i++) {
			//found a satisfied model and return it
			if(mc.PL_True(constraints, model)) return model;
			
			//haven't found a satisfied model
			if(randChoose(p)) {//flip a symbol in current model randomly
				model = randFlip(model);
			}
			else {//flip the symbol that can maximize the number of satisfied constrain
				maxSatFlip(constraints,model);
			}
		}
		
		//failure
		return null;
	}
	
	
	//to assign true/false for symbols and set them into model
	 private static List<Integer> randomModel(Set<Integer> symbols) {
		 Random rand = new Random();
		 Set<Integer> result = new HashSet<>();
		 
		 for (int symbol : symbols) {
			 // Randomly choose a sign: true for positive, false for negative
			 boolean positiveSign = rand.nextBoolean();
			 if (!positiveSign) {
				 symbol = -symbol;
			 }
	            result.add(symbol);
	        }
	        List<Integer> Tresult = new ArrayList<>(result);;
	        return Tresult;
	    }
	 
	 //to decide flip randomly or not by probability p
	 private boolean randChoose(double p) {
		 Random rand = new Random();
		 return rand.nextDouble() < p;
	 }
	 
	 //to get the model after flipping a random symbol
	 private List<Integer> randFlip(List<Integer> model) {
		 Random rand = new Random();
		 int n = rand.nextInt(model.size());
		 int flipedSym = -(model.get(n));
		 model.set(n, flipedSym);
		 return model;
	 }
	 
	 //to get the model after flipping the symbol that maximizes the number of satisfied constraints
	 private List<Integer> maxSatFlip(List<List<Integer>> constraints, List<Integer> model){
		 List<Integer> cur = model;
		 int cont = numOfTrue(constraints, model);
		 int cur_true;
		 int index_flip = -1;
		 Integer flip;
		 
		 for(int i = 0; i<model.size(); i++) {
			 flip = -(model.get(i));
			 cur.set(i, flip);
			 cur_true = numOfTrue(constraints, cur);
			 if(cur_true>cont) {
				 cont = cur_true;
				 index_flip = i;
			 }
			 cur = model;
		 }
		 
		 if(index_flip>-1){
			 flip = -(model.get(index_flip));
			 model.set(index_flip, flip);
		 }
		 return model;
	 }
	 //to get the number of constraints satisfied
	 private int numOfTrue(List<List<Integer>> constraints, List<Integer> model ) {
		 int cont = 0;
		 for(List<Integer> clause:constraints) {
			 for(Integer literal : clause) {
				 if(model.contains(literal)) {
					 cont++;
					 break;
				 }
			 }
		 }
			return cont;
		}
	 
	 //ask for p and max_tries and max_flips
	 private double GetP() {
		 Scanner scanner = new Scanner(System.in);
		 double p = -1;
		 do {
			 System.out.println("Please enter the probability(from 0 to 1):");
			 p = scanner.nextDouble();
		 }while((p<0.0)||(p>1.0));
		return p;
	 }
	 private int GetTries() {
		 Scanner scanner = new Scanner(System.in);
		 int max_tries = -1;
		 do {
			 System.out.println("Please enter the maximize tries:");
			 max_tries = scanner.nextInt();
		 }while(max_tries<0);
		return max_tries;
	 }
	 private int GetFlips() {
		 Scanner scanner = new Scanner(System.in);
		 int max_flips = -1;
		 do {
			 System.out.println("Please enter the maximize flips:");
			 max_flips = scanner.nextInt();
		 }while(max_flips<0);
		return max_flips;
	 }
	 
	 //print out
	 private void PrintBool(List<Integer> list) {
		 if(list == null) System.out.println("no result found.");
		 else {
			 System.out.print("[ ");
			 for(Integer i:list) {
				 if(i<0) {
					 System.out.print("F ");
				 }
				 else System.out.print("T ");
			 }
			 System.out.println("]");
		 }
	 }

}
