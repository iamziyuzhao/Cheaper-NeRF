/*
 * File: Distribution.java
 * Creator: George Ferguson
 */

package bn.base;

import java.util.Map;

import bn.core.RandomVariable;
import bn.core.Value;
import util.ArrayMap;

/**
 * Base implementation of a Distribution as an ArrayMap from Values to Doubles.
 * AIMA uses arrays indexed by the values of the variable. We use
 * a Map whose keys are the values. One could instead map the values
 * to integer indexes (by imposing an ordering on them), and then use
 * an array of doubles to represent the distribution. That might be
 * nice since one can't put primitive doubles in a Map...
 */
public class Distribution extends ArrayMap<Value,Double> implements bn.core.Distribution {
	
	protected RandomVariable variable;
	
	public RandomVariable getVariable() {
		return this.variable;
	}

	/**
	 * Construct and return a new empty Distribution for the given RandomVariable.
	 * Note that we don't actually enforce that the values in the Distribution
	 * are only those for the RandomVariable. But anyway, this is handy.
	 */
	public Distribution(RandomVariable X) {
		super(X.getRange().size());
		this.variable = X;
	}

	/**
	 * Set the probability for the given Value in this Distribution.
	 */
	@Override
	public void set(Value value, double probability) {
		put(value, Double.valueOf(probability));
	}

	/**
	 * Return the probability for the given Value in this Distribution.
	 */
	@Override
	public double get(Value value) {
		Double p = super.get(value);
		if (p == null) {
			throw new IllegalArgumentException(value.toString());
		} else {
			return p.doubleValue();
		}
	}

	/**
	 * Normalize this distribution so that the probabilities add up to 1.
	 */
	@Override
	public void normalize() {
		double sum = 0.0;
		for (Double value : values()) {
			sum += value.doubleValue();
		}
		// Avoid concurrent modification exceptions by modifying the entries directly
		for (Map.Entry<Value,Double> entry : this.entrySet()) {
			entry.setValue(entry.getValue()/sum); 
		}
	}
	
	/**
	 * Return a new copy of this Distribution that references the same RandomVariable
	 * and its Values, but has copies of the probabilities so that it can be
	 * changed without changing this Distribution.
	 * Note: ArrayMap.copy() doesn't copy the probabilities (Doubles).
	 */
	public Distribution copy() {
		Distribution newDist = new Distribution(this.variable);
		for (Map.Entry<Value,Double> entry : this.entrySet()) {
			Double oldProb = entry.getValue();
			Double newProb = Double.valueOf(oldProb);
			newDist.put(entry.getKey(), newProb);
		}
		return newDist;
	}

	/**
	 * Test Distributions.
	 */
	public static void main(String[] argv) {
		Value v1 = new StringValue("v1");
		Value v2 = new StringValue("v2");
		Value v3 = new StringValue("v3");
		Range range = new Range();
		range.add(v1);
		range.add(v2);
		range.add(v3);
		RandomVariable V = new NamedVariable("V", range); 
		Distribution dist = new Distribution(V);
		dist.set(v1, 0.4);
		dist.set(v2, 0.3);
		dist.set(v3, 0.1);
		System.out.println(dist);
		dist.normalize();
		System.out.println(dist);
	}
}
