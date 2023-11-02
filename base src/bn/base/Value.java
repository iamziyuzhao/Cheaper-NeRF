package bn.base;

/**
 * Base implementation of Values allows you to wrap any other type and
 * make instance of that type into Values.
 */
public class Value<T> implements bn.core.Value {

	protected T value;
	
	/**
	 * Return the value of this Value, if you follow me.
	 */
	public T getStringValue() {
		return value;
	}

	/**
	 * Construct and return a new Value using the given value.
	 */
	public Value(T value) {
		this.value = value;
	}

	/**
	 * Return the String representation of this Value.
	 */
	public String toString() {
		return value.toString();
	}
	
	/**
	 * Return true if the given object is a StringValue (not null) and
	 * its value (a String) is String.equals to this StringValue's value
	 * (which is also a String).
	 * Whew!
	 */
	public boolean equals(Object other) {
		if (other != null && other instanceof Value) {
			try {
				@SuppressWarnings("unchecked")
				Value<T> tother = (Value<T>)other;
				return (this.value.equals(tother.value));
			} catch (ClassCastException ex) {
				return false;
			}
		} else {
			return false;
		}
	}
	
	/**
	 * The hashCode of a Value is the hashCode of its value.
	 */
	public int hashCode() {
		return value.hashCode();
	}
}
