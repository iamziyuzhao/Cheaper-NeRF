package bn.base;

/**
 * A BooleanRange is a Range containing the two BooleanValues TRUE and FALSE.
 */
public class BooleanRange extends Range {
	
	public BooleanRange() {
		super(2);
		this.add(BooleanValue.TRUE);
		this.add(BooleanValue.FALSE);
	}

}
