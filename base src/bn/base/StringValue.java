package bn.base;

/**
 * A StringValue is a String that can be used as a Value.
 */
public class StringValue extends Value<String> {

	/**
	 * Construct and return a new StringValue using the given String.
	 */
	public StringValue(String s) {
		super(s);
	}
	
	/**
	 * Return the String value of this StringValue, if you follow me.
	 */
	public String stringValue() {
		return this.value;
	}

	/**
	 * Test StringValues.
	 */
	public static void main(String[] argv) {
		StringValue red = new StringValue("red");
		System.out.println(red);
	}
	
}
