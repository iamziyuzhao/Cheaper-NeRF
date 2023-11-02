package bn.core;

import java.util.Map;

/**
 * A Distribution for a RandomVariable is a probability for each of its Values
 * such that the values satisfy the axioms of probability (are all between
 * 0 and 1, inclusive, and sum to 1).
 */
public interface Distribution extends Map<Value,Double> {
	
	public RandomVariable getVariable();
	
	public void set(Value value, double probability);

	public double get(Value value);

	public void normalize();

}
