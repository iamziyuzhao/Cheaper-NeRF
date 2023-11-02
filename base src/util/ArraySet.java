/*
 * File: ArraySet.java
 * Creator: George Ferguson
 * Created: Tue Mar 15 16:33:49 2011
 * Time-stamp: <Thu Mar 15 12:20:55 EDT 2012 ferguson>
 */

package util;

import java.util.AbstractSet;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;

/**
 * A Set implementation backed by an ArrayList.
 * This is good for small, more or less immutable sets, but isn't part of
 * standard Java.
 * Note that the order of iteration in an ArraySet is going to be the same
 * as for an ArrayList, which is in the order elements are added. But of
 * course that isn't guaranteed for Sets in general.
 */
public class ArraySet<E> extends AbstractSet<E> {

	protected ArrayList<E> elements;

	/**
	 * Construct and return a new ArraySet with initial size 0.
	 */
	public ArraySet() {
		this(0);
	}

	/**
	 * Construct and return a new ArraySet with the given
	 * initial capacity.
	 */
	public ArraySet(int initialCapacity) {
		super();
		elements = new ArrayList<E>(initialCapacity);
	}

	/**
	 * Construct and return a new ArraySet containing the elements
	 * of the given collection, with initial capacity as needed to hold
	 * them all.
	 * Per the contract of the Set interface in the Collections
	 * Framework, duplicate elements (according to equals()) are not added
	 * multiple times.
	 */
	public ArraySet(Collection<? extends E> elements) {
		this(elements.size());
		for (E e : elements) {
			add(e);
		}
	}

	/**
	 * Construct and return a new ArraySet containing the given elements
	 * or array of elements, with initial capacity sufficient to hold
	 * them all.
	 * Duplicate elements are not added more than once.
	 */
	@SafeVarargs
	public ArraySet(E... elements) {
		this(elements.length);
		for (E e : elements) {
			add(e);
		}
	}

	/**
	 * Returns an iterator over the elements in this ArraySet.
	 * Required by subclasses of AbstractSet.
	 */
	@Override
	public Iterator<E> iterator() {
		return elements.iterator();
	}

	/**
	 * Returns the number of elements in this ArraySet (its cardinality).
	 * Hopefully this is faster than letting AbstractSet compute it using the
	 * iterator.
	 */
	@Override
	public int size() {
		return elements.size();
	}

	/**
	 * Adds the specified element to this ArraySet if it is not already present.
	 * Returns true if the element was added, else false.
	 * Required by AbstractSet for a mutable set. It could perhaps be
	 * implemented more efficiently...
	 */
	@Override
	public boolean add(E e) {
		if (contains(e)) {
			return false;
		} else {
			return elements.add(e);
		}
	}
	
	/**
	 * Returns the element at the given index in this ArraySet.
	 * I know Sets aren't necessarily ordered, but these ones are.
	 */
	public E get(int index) {
		return elements.get(index);
	}

	/**
	 * Return a new ArraySet containing the same elements as this one (so a
	 * ``shallow'' copy).
	 * This can be done in linear time since we know that we don't have
	 * duplicate elements, so can bypass ArraySet.add() which needs to check.
	 */
	public ArraySet<E> copy() {
		ArraySet<E> newSet = new ArraySet<E>(this.size());
		for (E e : this) {
			// Add directly to list of elements for the copy
			newSet.elements.add(e);
		}
		return newSet;
	}
}
