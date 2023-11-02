package bn.base;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Set;

import bn.core.Assignment;
import bn.core.CPT;
import bn.core.Named;
import bn.core.RandomVariable;
import bn.core.Value;
import util.ArraySet;

/**
 * Base implementation of a BayesianNetwork as a graph of Nodes
 * each of which corresponds to a RandomVariable with a CPT.
 * @see bn.core.BayesianNetwork
 * @see bn.core.RandomVariable
 * @see bn.core.CPT
 */
public class BayesianNetwork implements bn.core.BayesianNetwork, Cloneable {

	public class Node {

		RandomVariable variable;
		Set<Node> parents;
		Set<Node> children = new ArraySet<Node>();
		CPT cpt;

		Node(RandomVariable variable) {
			this.variable = variable;
		}
	}

	/**
	 * Construct and return a new (empty) BayesianNetwork initialized to contain
	 * the given number of RandomVariables. This is used to pre-allocate for
	 * storing them, which may be more efficient if you know the size in advance.
	 */
	public BayesianNetwork(int size) {
		this.nodes = new ArraySet<Node>(size);
	}
	
	/**
	 * Construct and return a new (empty) BayesianNetwork.
	 */
	public BayesianNetwork() {
		this(0);
	}
	
	/**
	 * Construct and return a new (empty) BayesianNetwork initialized to contain
	 * the given Collection (List, Set, ...) of RandomVariables.
	 */
	public BayesianNetwork(Collection<RandomVariable> variables) {
		this(variables.size());
		for (RandomVariable var : variables) {
			this.add(var);
		}
	}
	
	// Graph nodes

	/**
	 * The Set of Nodes representing this BayesianNetwork.
	 * We use an ArraySet since once we have created the network, it
	 * never changes. 
	 */
	protected Set<Node> nodes;

	/**
	 * Return the size (number of nodes == number of RandomVariables) in
	 * this BayesianNetwork.
	 */
	public int size() {
		return nodes.size();
	}

	/**
	 * Add a node for the given RandomVariable to this BayesianNetwork.
	 * Note: This is called when a {@code variable} entity is read in an XMLBIF
	 * file.
	 */
	public void add(RandomVariable var) {
		nodes.add(new Node(var));
	}

	/**
	 * Return the Node for given RandomVariable from this BayesianNetwork.
	 * @throws NoSuchElementException
	 */
	public Node getNodeForVariable(RandomVariable var) {
		for (Node node : nodes) {
			if (node.variable == var) {
				return node;
			}
		}
		throw new NoSuchElementException();
	}

	/**
	 * Return the RandomVariable with the given name from this BayesianNetwork.
	 * Only tests variables that implement the Named interface.
	 * <p>
	 * This is currently only used when reading a network from an XMLBIF
	 * file. If you used it more heavily, you'd probably want to index
	 * the variables more efficiently.
	 * @throws NoSuchElementException
	 */
	@Override
	public RandomVariable getVariableByName(String name) {
		for (Node node : nodes) {
			RandomVariable var = node.variable;
			if (Named.class.isInstance(var)) {
				NamedVariable nvar = (NamedVariable)var;
				if (nvar.getName().equals(name)) {
					return var;
				}
			}
		}
		throw new NoSuchElementException();
	}

	/**
	 * Return the RandomVariables in this BayesianNetwork as a List.
	 * Current implementation uses an ArrayList for this, and the order of
	 * the variables will be the same as the order they were added
	 * to the network.
	 */
	@Override
	public Set<RandomVariable> getVariables() {
		Set<RandomVariable> vars = new ArraySet<RandomVariable>(nodes.size());
		for (Node node : nodes) {
			vars.add(node.variable);
		}
		return vars;
	}
	
	// Graph edges

	/**
	 * Connect the node for the given RandomVariable to the nodes for
	 * the given set of parent RandomVariables, with the given CPT.
	 * Note: This is called when a {@code definition} entity is read in an XMLBIF
	 * file.
	 */
	public void connect(RandomVariable var, Set<RandomVariable> parents, CPT cpt) {
		Node node = getNodeForVariable(var);
		node.parents = new ArraySet<Node>(parents.size());
		for (RandomVariable pvar : parents) {
			Node pnode = getNodeForVariable(pvar);
			node.parents.add(pnode);
			pnode.children.add(node);
		}
		node.cpt = cpt;
	}

	/**
	 * Returns the Set of RandomVariables that are the children of
	 * the given RandomVariable. This is done in a really wasteful
	 * way, so you'd be better off doing something different for
	 * use in a certain kind of sampling...
	 */
	@Override
	public Set<RandomVariable> getChildren(RandomVariable X) {
		//trace("BayesianNetwork.getChildren: X=" + X);
		Set<RandomVariable> children = new ArraySet<RandomVariable>();
		Node node = getNodeForVariable(X);
		for (Node childNode: node.children) {
			children.add(childNode.variable);
			//trace("BayesianNetwork.getChildren: " + childNode.variable);
		}
		return children;
	}

	/**
	 * Returns the Set of RandomVariables that are the parents of
	 * the given RandomVariable. Also done inefficiently if you are
	 * calling it frequently.
	 */
	@Override
	public Set<RandomVariable> getParents(RandomVariable X) {
		Set<RandomVariable> parents = new ArraySet<RandomVariable>();
		Node node = getNodeForVariable(X);
		for (Node parentNode: node.parents) {
			parents.add(parentNode.variable);
		}
		return parents;
	}
	
	// CPT lookup

	/**
	 * Return the probability stored in the CPT for the given RandomVariable,
	 * given the Values of its parents (and itself) in the given Assignment.
	 */
	public double getProbability(RandomVariable X, Assignment e) {
		//trace("BayesianNetwork.getProb: for variable " + X + ", e=" + e);
		Node node = getNodeForVariable(X);
		Value value = e.get(X);
		double result = node.cpt.get(value, e);
		//trace("BayesianNetwork.getProb: result=" + result);
		return result;
	}
	
	/**
	 * Set the probability stored in the CPT for the given RandomVariable,
	 * given the Values of its parents (and itself) in the given Assignment.
	 */
	public void setProbability(RandomVariable X, Assignment e, double p) {
		//trace("BayesianNetwork.setProb: for variable " + X + ", e=" + e + ": " + p);
		Node node = getNodeForVariable(X);
		Value value = e.get(X);
		node.cpt.set(value, e, p);
	}
	
	// Topsort variables

	/**
	 * Sort the given list of RandomVariables topologically with respect
	 * to this BayesianNetwork, and return the resulting list.
	 * I thought about just having a {@code topsort} method, into which
	 * one would pass a collection of RandomVariables (e.g., from
	 *{@link BayesianNetwork#getVariables}), but there was just too much
	 * converting between RandomVariables and Nodes for that. The only
	 * reason you ever topsort is to topsort the entire list of variables
	 * once at the start of inference, so why not just do that here.
	 * <p>
	 * This implementation uses the DFS algorithm described in Wikipedia
	 * and attributed to Tarjan.
	 */
	public List<RandomVariable> getVariablesSortedTopologically() {
		// ``L <- Empty list that will contain the sorted nodes''
		List<RandomVariable> L = new ArrayList<RandomVariable>(nodes.size());
		// ``S <- Set of all nodes with no outgoing edges''
		Set<Node> S = new ArraySet<Node>(nodes.size());
		for (Node node : nodes) {
			if (node.children.isEmpty()) {
				S.add(node);
			}
		}
		// Can't mark nodes visited; instead keep as a set
		Set<Node> visited = new ArraySet<Node>(nodes.size());
		// ``for each node n in S do''
		for (Node n : S) {
			// ``visit(n)''
			visit(n, L, visited);
		}
		return L;
	}

	/**
	 * RecursiveNoCycleCheck step of topological sort procedure.
	 */
	protected void visit(Node n, List<RandomVariable> L, Set<Node> visited) {
		// ``if n has not been visited yet''
		if (!visited.contains(n)) {
			// ``mark n as visited''
			visited.add(n);
			// ``for each node m with an edge from m to n do''
			for (Node m : nodes) {
				if (m.children.contains(n)) {
					// ``visit(m)''
					visit(m, L, visited);
				}
			}
			// ``add n to L''
			L.add(n.variable);
		}
	}
	
	// Printing
	
	/**
	 * Return the String representation of this BayesianNetwork.
	 */
	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		for (RandomVariable var : getVariablesSortedTopologically()) {
			builder.append(var.toString());
			builder.append(" <- ");
			Node node = getNodeForVariable(var);
			if (node.parents != null) {
				for (Node pnode : node.parents) {
					builder.append(pnode.variable.toString());
					builder.append(" ");
				}
			}
			builder.append("\n");
			if (node.cpt != null) {
				// Might not want this if it clutters things up...
				builder.append(node.cpt.toString());
				builder.append("\n");
			}
		}
		return builder.toString();
	}
	
	/**
	 * Return a copy of this BayesianNetwork that uses the same RandomVariables,
	 * but has its own copies of the CPTs (initialized to this network's values).
	 * This method is used by some learning algorithms, such as EM. 
	 */
	@Override
	public BayesianNetwork copy() {
		Set<RandomVariable> variables = this.getVariables();
		// New network with the same variables
		BayesianNetwork newNetwork = new BayesianNetwork(variables);
		// Copy the topology
		for (RandomVariable var : variables) {
			Node node = this.getNodeForVariable(var);
			Set<Node> parents = node.parents;
			Set<RandomVariable> newParents = new ArraySet<RandomVariable>(parents.size());
			for (Node parentNode : parents) {
				RandomVariable parentVar = parentNode.variable;
				newParents.add(parentVar);
			}
			// Copy CPT
			CPT newCPT = node.cpt.copy();
			// Connect
			newNetwork.connect(var, newParents, newCPT);
		}
		return newNetwork;
	}
	
	public static void main(String[] argv) {
		BayesianNetwork network = new BayesianNetwork();
		RandomVariable A = new NamedVariable("A", new BooleanRange());
		RandomVariable B = new NamedVariable("B", new BooleanRange());
		network.add(A);
		network.add(B);

		// Shorthands
		BooleanValue TRUE = BooleanValue.TRUE;
		BooleanValue FALSE = BooleanValue.FALSE;
		Assignment a;

		// A (no parents)
		CPT Aprior = new bn.base.CPT(A);
		a = new bn.base.Assignment();
		Aprior.set(TRUE, a, 0.6);
		Aprior.set(FALSE, a, 0.4);
		network.connect(A, new ArraySet<RandomVariable>() , Aprior);

		// A -> B
		Set<RandomVariable> justA = new ArraySet<RandomVariable>();
		justA.add(A);
		CPT BgivenA = new bn.base.CPT(B);
		a = new bn.base.Assignment();
		a.put(A, TRUE);
		BgivenA.set(TRUE, a, 0.8);
		BgivenA.set(FALSE, a, 1-0.8);
		a = new bn.base.Assignment();
		a.put(A, FALSE);
		BgivenA.set(TRUE, a, 0.4);
		BgivenA.set(FALSE, a, 1-0.4);
		network.connect(B, justA, BgivenA);

		System.out.println(network);
		
		BayesianNetwork network2 = network.copy();
		System.out.println(network2);
		
	}
	
}
