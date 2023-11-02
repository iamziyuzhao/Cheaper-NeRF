package bn.base;

import java.util.Collection;

import bn.core.Value;
import util.ArraySet;

/**
 * Base implementation of a Range as an ArraySet of Values. 
 * We use a ArraySet here since iteration is the main use for Ranges
 * in Bayesian network algorithms.
 */
public class Range extends ArraySet<Value> implements bn.core.Range {

	public static final long serialVersionUID = 1L;

	public Range() {
		super();
	}

	public Range(int size) {
		super(size);
	}

	public Range(Value... values) {
		this();
		for (Value value : values) {
			this.add(value);
		}
	}

	public Range(Collection<Value> collection) {
		this();
		for (Value value : collection) {
			this.add(value);
		}
	}

	public static void main(String[] argv) {
		StringValue red = new StringValue("red");
		StringValue green = new StringValue("green");
		StringValue blue = new StringValue("blue");
		Range range = new Range();
		range.add(red);
		range.add(green);
		range.add(blue);
		System.out.println(range);
		Range booleans = new Range(BooleanValue.TRUE, BooleanValue.FALSE);
		System.out.println(booleans);
	}
}
