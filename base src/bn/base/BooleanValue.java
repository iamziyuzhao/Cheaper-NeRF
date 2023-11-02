package bn.base;

/**
 * A BooleanValue is a Boolean that can be used as a Value.
 */
public class BooleanValue extends Value<Boolean> {
	
	/**
	 * Construct and return a new BooleanValue using the given Boolean.
	 */
	public BooleanValue(Boolean b) {
		super(b);
	}
	
	/**
	 * Return the Boolean value of this BooleanValue, if you follow me.
	 * Note that this constructor is protected to prevent new instances
	 * other than TRUE and FALSE from being created.
	 */
	protected Boolean booleanValue() {
		return this.value;
	}

	/**
	 * The constant BooleanValue for Boolean.TRUE.
	 */
	public static final BooleanValue TRUE = new BooleanValue(Boolean.TRUE);

	/**
	 * The constant BooleanValue for Boolean.FALSE.
	 */
	public static final BooleanValue FALSE = new BooleanValue(Boolean.FALSE);

	/**
	 * Returns the BooleanValue TRUE if the given boolean is true, otherwise
	 * FALSE if it false.
	 */
	public static BooleanValue valueOf(boolean b) {
		if (b) {
			return TRUE;
		} else {
			return FALSE;
		}
	}

	/**
	 * Returns the BooleanValue TRUE if the given String is "true" (ignoring
	 * case), FALSE if the given string is "false" (ditto), and throws an
	 * IllegalArgumentException for anything else.
	 * Note that Boolean.valueOf returns TRUE for "true" and FALSE for anything
	 * else. Why would they do that?
	 */
	public static BooleanValue valueOf(String s) throws IllegalArgumentException {
		s = s.toLowerCase();
		if (s.equals("true") ) {
			return TRUE;
		} if (s.equals("false")) {
			return FALSE;
		} else {
			throw new IllegalArgumentException(s);
		}
	}
	
	/**
	 * Test BooleanValues.
	 */
	public static void main(String[] argv) {
		BooleanValue true1 = BooleanValue.valueOf(true);
		BooleanValue false1 = BooleanValue.valueOf(false);
		BooleanValue true2 = BooleanValue.valueOf("true");
		BooleanValue false2 = BooleanValue.valueOf("FALSE");
		System.out.println("true1=" + true1);
		System.out.println("false1=" + false1);
		System.out.println("true2=" + true2);
		System.out.println("false2=" + false2);
		System.out.println("true1 equals true1? " +true1.equals(true1));
		System.out.println("true1 equals false1? " + true1.equals(false1));
		System.out.println("true1 equals true2?  " + true1.equals(true2));
		System.out.println("false1 equals false2? " + false1.equals(false2));
		try {
			BooleanValue broken = BooleanValue.valueOf("hello");
			System.out.println(broken);
		} catch (IllegalArgumentException e) {
			System.out.println(e);
		}
	}
	
}
