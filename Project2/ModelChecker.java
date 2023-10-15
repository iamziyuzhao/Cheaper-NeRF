import java.util.*;

public class ModelChecker {

    public ModelChecker()
    {
    }

    public boolean ttEntails(List<List<Integer>> kb, List<List<Integer>> alpha)
    {
        Set<Integer> symbols = getSymbols(kb);
        // printSet(symbols);
        return ttCheckAll(kb, alpha, symbols, new ArrayList<>());
    }

    private void printSet(Set<Integer> set)
    {
        for (Integer num : set)
        {
            System.out.print(num + " ");
        }
        System.out.println();
    }

    public boolean ttCheckAll(List<List<Integer>> kb, List<List<Integer>> alpha, Set<Integer> symbols, List<Integer> model)
    {
        if (symbols.isEmpty())
        {
            if (PL_True(kb, model))
                return PL_True(alpha, model);
            else
                return true;
        }
        else
        {
            Integer P = symbols.iterator().next();
            symbols.remove(P);

            List<Integer> model1= new ArrayList<>(model);
            model1.add(P);
            List<Integer> model2= new ArrayList<>(model);
            model2.add(P*-1);

            return (ttCheckAll(kb, alpha, symbols, model1)) &&
                   (ttCheckAll(kb, alpha, symbols, model1));
        }
    }

    protected Set<Integer> getSymbols(List<List<Integer>> clauses)
    {
        Set<Integer> symbols = new HashSet<>();

        for (List<Integer> clause : clauses)
            for (int num : clause)
                symbols.add(Math.abs(num));

        return symbols;
    }

    protected boolean PL_True(List<List<Integer>> clauses, List<Integer> model ) {
		for(List<Integer> clause:clauses) {
			boolean clause_true = false;
			for(Integer literal : clause) {
				if(model.contains(literal)) {
					clause_true = true;
					break;
				}
			}
			if(clause_true == false) {
				return false;
			}
		}
		return true;
	}
}
