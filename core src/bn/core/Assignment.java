package bn.core;

import java.util.Map;
import java.util.Set;

/**
 * An Assignment is mapping (a set of assignments) from RandomVariables to
 * Values from their Ranges.
 */
public interface Assignment extends Map<RandomVariable,Value> {
	
	/**
	 * Return a Set view of the RandomVariables in this Assignment.
	 * @see Map.keySet()
	 */
	public Set<RandomVariable> variableSet();
	
	/**
	 * Return true if this Assignment contains all the assignments
	 * in the given other Assignment. That is, the other Assignment is
	 * a subset of this one (or they are equal).
	 */
	public boolean containsAll(Assignment other);
	
	/**
	 * Return a shallow copy of this Assignment (that is, an Assignment that
	 * contains the same assignments without copying the RandomVariables or
	 * Values).
	 */
	public Assignment copy();

}
