package bn.core;

/**
 * A RandomVariable can be assigned a Value from its Range.
 * Note that RandomVariables may be used as keys in a hashtable, so they should
 * have appropriate hashCode() methods.
 */
public interface RandomVariable {
	
	public Range getRange();

}
