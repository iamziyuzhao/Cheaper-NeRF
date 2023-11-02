package bn.base;

import bn.core.Range;

/**
 * BooleanVariables all shared a single instance of a BooleanRange.
 */
public class BooleanVariable extends NamedVariable {

	static Range range = new BooleanRange();

	public BooleanVariable(String name) {
		super(name, range);
	}

}
