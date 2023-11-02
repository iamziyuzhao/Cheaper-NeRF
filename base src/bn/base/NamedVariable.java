package bn.base;

import bn.core.Range;
import bn.core.Named;
import bn.core.RandomVariable;

/**
 * Base implementation of a RandomVariable that has a name as well
 * as its Range of Values.
 */
public class NamedVariable implements RandomVariable, Named {
	
	protected String name;
	protected Range range;
	
	public NamedVariable(String name, Range range) {
		this.name = name;
		this.range = range;
	}
	
	@Override
	public String getName() {
		return this.name;
	}
	
	@Override
	public Range getRange() {
		return this.range;
	}
	
	@Override
	public String toString() {
		return this.name;
	}
	
	/*
	 * Not sure this is needed since we don't actually compare RandomVariables
	 * to each other using equals(). So the default hashCode() implementation
	 * may be sufficient.
	@Override
	public int hashCode() {
		return this.name.hashCode();
	}
	 */
	
	// Testing
	
	public static void main(String[] argv) {
		StringValue red = new StringValue("red");
		StringValue green = new StringValue("green");
		StringValue blue = new StringValue("blue");
		bn.base.Range range = new bn.base.Range();
		range.add(red);
		range.add(green);
		range.add(blue);
		NamedVariable v1 = new NamedVariable("Color", range);
		System.out.format("%s : %s\n", v1, v1.getRange());
		bn.base.Range booleans = new bn.base.Range(BooleanValue.TRUE, BooleanValue.FALSE);
		NamedVariable v2 = new NamedVariable("IsFurry", booleans);
		System.out.format("%s : %s\n", v2, v2.getRange());
	}

}
