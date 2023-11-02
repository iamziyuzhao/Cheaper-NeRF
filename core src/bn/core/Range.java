package bn.core;

import java.util.Iterator;

/**
 * The Range of a RandomVariable is a finite set of Values (at least for CSC242).
 * <p>
 * This was called the “domain” of a random variable in AIMA 3rd ed. (presumably
 * to go with the domains of variables in CSPs). However since random variables
 * are functions from possible worlds to values, it makes sense to refer to
 * their “range” of values (the “range” of a function).,
 * <p>
 * This is a subset of the Set interface (we could just extend Set\&lt;Value&gt;
 * instead). Note that in the Set interface, contains() and remove() take
 * Objects as parameters, rather than whatever the element type is (Value in
 * this case). Not sure why that is...
 * @see https://stackoverflow.com/questions/104799/why-arent-java-collections-remove-methods-generic
 */
public interface Range extends Iterable<Value> {
	
	public boolean add(Value value);
	
	public boolean contains(Object obj);
	
	public boolean remove(Object obj);
	
	public void clear();
	
	public boolean isEmpty();
	
	public int size();
	
	public Iterator<Value> iterator();

}
