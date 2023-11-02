package bn.core;

import java.util.List;
import java.util.Set;

/**
 * A BayesianNetwork is a directed acyclic graph where the nodes
 * correspond to RandomVariables and store the conditional distribution
 * of their variable given its parents.
 */
public interface BayesianNetwork {
	
	/**
	 * Add a node for the given RandomVariable to this BayesianNetwork.
	 * Note: This is called when a {@code variable} entity is read in an XMLBIF
	 * file.
	 */
	public void add(RandomVariable var);
	
	/**
	 * Connect the node for the given RandomVariable to the nodes for
	 * the given set of parent RandomVariables, with the given CPT.
	 * Note: This is called when a {@code definition} entity is read in an XMLBIF
	 * file.
	 */
	public void connect(RandomVariable var, Set<RandomVariable> parents, CPT cpt);
	
	/**
	 * Return the RandomVariables used in this BayesianNetwork.
	 */
	public Set<RandomVariable> getVariables();

	/**
	 * Return the RandomVariables used in this BayesianNetwork sorted
	 * topologically (so that parents precede their children).
	 */
	public List<RandomVariable> getVariablesSortedTopologically();

	/**
	 * Return the RandomVariables that are parents of the given RandomVariable.
	 */
	public Set<RandomVariable> getParents(RandomVariable var);

	/**
	 * Return the RandomVariables that are Children of the given RandomVariable.
	 */
	public Set<RandomVariable> getChildren(RandomVariable var);
	
	/**
	 * Return the probability stored in the CPT for the given RandomVariable,
	 * given the Values of its parents (and of itself) in the given Assignment.
	 */
	public double getProbability(RandomVariable X, Assignment e);
	
	/**
	 * Set the probability stored in the CPT for the given RandomVariable,
	 * given the Values of its parents (and of itself) in the given Assignment.
	 */
	public void setProbability(RandomVariable X, Assignment e, double p);
	
	/**
	 * Return the RandomVariable with the given name from this BayesianNetwork.
	 */
	public RandomVariable getVariableByName(String name);
	
	/**
	 * Return a copy of this BayesianNetwork that uses the same RandomVariables,
	 * but has its own copies of the CPTs (initialized to this network's values).
	 * <p>
	 * BayesianNetworks must support the copy() operation for some learning
	 * algorithms, such as EM. If you don't use those algorithms, implement this
	 * method and throw an UnsupportedOperationException.
	 */
	public BayesianNetwork copy();
	
}
