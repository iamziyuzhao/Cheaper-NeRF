package bn.examples;

import java.util.Set;

import bn.base.BayesianNetwork;
import bn.base.BooleanRange;
import bn.base.BooleanValue;
import bn.base.NamedVariable;
import bn.base.Range;
import bn.base.StringValue;
import bn.core.Assignment;
import bn.core.CPT;
import bn.core.RandomVariable;
import bn.inference.EnumerationInferencer;
import util.ArraySet;

/**
 * The very simple BayesianNetwork from the CSC242 Project 3 description,
 * with some probabilities made up for the CPTs. 
 * This network is small enough that you can EASILY compute the answer
 * to any query manually in order to check the performance of your BN inference
 * implementations.
 */
public class Project3SmallNetwork extends BayesianNetwork {
	
	public final StringValue high = new StringValue("high");
	public final StringValue med = new StringValue("med");
	public final StringValue low = new StringValue("low");

	public final RandomVariable A = new NamedVariable("A", new BooleanRange());
	public final RandomVariable B = new NamedVariable("B", new BooleanRange());
	public final RandomVariable C = new NamedVariable("C", new Range(high, med, low));

	public Project3SmallNetwork() {
		super();

		this.add(A);
		this.add(B);
		this.add(C);

		// Shorthands
		BooleanValue TRUE = BooleanValue.TRUE;
		BooleanValue FALSE = BooleanValue.FALSE;

		Assignment assignment;

		// A (no parents): P(A) = <0.6, 0.4>
		Set<RandomVariable> parents = new ArraySet<RandomVariable>();	// none
		CPT Aprior = new bn.base.CPT(A);
		assignment = new bn.base.Assignment();
		Aprior.set(TRUE, assignment, 0.6);								// A true
		Aprior.set(FALSE, assignment, 0.4);								// A false
		this.connect(A, parents, Aprior);

		// A -> B: P(B|A) = <a=t: <0.2, 0.8>, a=f: <0.9, 0.1>>
		parents = new ArraySet<RandomVariable>(A);				// just A
		CPT BgivenA = new bn.base.CPT(B);
		assignment = new bn.base.Assignment();
		assignment.put(A, TRUE);
		BgivenA.set(TRUE, assignment, 0.2);						// A true, B true
		BgivenA.set(FALSE, assignment, 0.8);						// A true, B false
		assignment = new bn.base.Assignment();
		assignment.put(A, FALSE);
		BgivenA.set(TRUE, assignment, 0.9);						// A false, b true
		BgivenA.set(FALSE, assignment, 0.1);						// A false, b false
		this.connect(B, parents, BgivenA);

		// A -> C: P(C|A) = <a=t: <0.1, 0.3, 0.6>, a=f: <0.2, 0.5, 0.3>>
		parents = new ArraySet<RandomVariable>(A);				// just A
		CPT CgivenA = new bn.base.CPT(C);
		assignment = new bn.base.Assignment();
		assignment.put(A, TRUE);
		CgivenA.set(high, assignment, 0.1);						// A true, C high
		CgivenA.set(med, assignment, 0.3);						// A true, B med
		CgivenA.set(low, assignment, 0.6);						// A true, B low
		assignment = new bn.base.Assignment();
		assignment.put(A, FALSE);
		CgivenA.set(high, assignment, 0.2);						// A false, C high
		CgivenA.set(med, assignment, 0.5);						// A false, B med
		CgivenA.set(low, assignment, 0.3);						// A false, B low
		this.connect(C, parents, CgivenA);
	}

	public static void main(String[] args) {
		var bn = new Project3SmallNetwork();
		System.out.println(bn);

		// You need to provide this class!
		var exact = new EnumerationInferencer();
		
		var e = new bn.base.Assignment();
		e.put(bn.B, BooleanValue.TRUE);							// B true
		e.put(bn.C, bn.high);									// C high
		// P(A|b,high) = alpha*<P(a)P(b|a)P(high|a), P(~a)P(b|~a)P(high|~a)> (no hidden vars)
		//             = alpha*<0.6*0.2*0.1, 0.4*0.9*0.2>
		//             = alpha*<0.012, 0.072> = <0.142857142857143, 0.857142857142857>
		System.out.println("P(A|b,high) = alpha*<0.012, 0.072> = <0.142857142857143, 0.857142857142857>");
		var dist = exact.query(bn.A, e, bn);
		System.out.println(dist);

		e = new bn.base.Assignment();
		e.put(bn.C, bn.med);										// C med
		// P(B|med) = alpha*<P(a)P(b|a)P(med|a) + P(~a)P(b|~a)P(med|~a),
		//					 P(a)P(~b|a)P(med|a) + P(~a)P(~b|~a)P(med|~a)> 
		//             = alpha*<0.6*0.2*0.3 + 0.4*0.9*0.5, 0.6*0.8*0.3 + 0.4*0.1*0.5>
		//             = alpha*<0.216, 0.164> = <0.568421052631579, 0.431578947368421>
		System.out.println("P(B|high) = alpha*<0.216, 0.164> = <0.568421052631579, 0.431578947368421>");
		dist = exact.query(bn.B, e, bn);
		System.out.println(dist);
	}
}
