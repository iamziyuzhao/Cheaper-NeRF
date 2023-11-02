/*
 * File: XMLBIFParser.java
 * Creator: George Ferguson
 */

package bn.parser;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.StringTokenizer;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import bn.core.Assignment;
import bn.core.BayesianNetwork;
import bn.core.CPT;
import bn.core.RandomVariable;
import bn.core.Range;
import bn.core.Value;

/**
 * DocumentBuilder-based DOM parser for
 * <a href="http://sites.poli.usp.br/p/fabio.cozman/Research/InterchangeFormat/index.html">XMLBIF</a>
 * files.
 * <p>
 * Note that XMLBIF explicitly states that <q>There is no mandatory
 * order of variable and probability blocks.</q> This means that we
 * have to read the DOM, then create nodes for all the variables using
 * the {@code variable} elements, then hook them up and add the CPTs
 * using the {@code definition} blocks. A good reason to use a DOM
 * parser rather than a SAX parser.
 * <p>
 * Also XMLBIF appears to use uppercase tag names, perhaps thinking they
 * really ought to be case-insensitive.
 * <p>
 * I have implemented minimal sanity checking and error handling.
 * You could do better. Caveat codor.
 */
public class XMLBIFParser {

	public BayesianNetwork readNetworkFromFile(String filename) throws IOException, ParserConfigurationException, SAXException {
		DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
		DocumentBuilder db = dbf.newDocumentBuilder();
		Document doc = db.parse(new File(filename));
		return processDocument(doc);
	}

	protected BayesianNetwork processDocument(Document doc) {
		final BayesianNetwork network = new bn.base.BayesianNetwork();
		// First do the variables
		doForEachElement(doc, "VARIABLE", new ElementTaker() {
			public void element(Element e) {
				processVariableElement(e, network);
			}
		});
		// Then do the defintions (a.k.a, links and CPTs)
		doForEachElement(doc, "DEFINITION", new ElementTaker() {
			public void element(Element e) {
				processDefinitionElement(e, network);
			}
		});
		return network;
	}

	protected void doForEachElement(Document doc, String tagname, ElementTaker taker) {
		NodeList nodes = doc.getElementsByTagName(tagname);
		if (nodes != null && nodes.getLength() > 0) {
			for (int i=0; i < nodes.getLength(); i++) {
				Node node = nodes.item(i);
				taker.element((Element)node);
			}
		}
	}

	protected void processVariableElement(Element e, BayesianNetwork network) {
		Element nameElt = getChildWithTagName(e, "NAME");
		String name = getChildText(nameElt);
		//trace("processing variable: " + name);
		final Range range = new bn.base.Range();
		doForEachChild(e, "OUTCOME", new ElementTaker() {
			public void element(Element e) {
				String value = getChildText(e);
				//trace("  adding value: " + value);
				// All values are strings in the XML; some of them should probably be Boolean...
				range.add(new bn.base.StringValue(value));
			}
		});
		RandomVariable var = new bn.base.NamedVariable(name, range);
		network.add(var);
	}

	protected void processDefinitionElement(Element e, final BayesianNetwork network) {
		Element forElt = getChildWithTagName(e, "FOR");
		String forName = getChildText(forElt);
		//trace("processing definition for: " + forName);
		RandomVariable forVar = network.getVariableByName(forName);
		// Have to preserve order of parents (``givens'') since that's used in CPT (``TABLE'')
		final List<RandomVariable> givens = new ArrayList<RandomVariable>();
		doForEachChild(e, "GIVEN", new ElementTaker() {
			public void element(Element e) {
				String value = getChildText(e);
				//trace("  adding parent: " + value);
				givens.add(network.getVariableByName(value));
			}
		});
		CPT cpt = new bn.base.CPT(forVar);
		Element tableElt = getChildWithTagName(e, "TABLE");
		String tableStr = getChildText(tableElt);
		initCPTFromString(cpt, givens, tableStr);
		Set<RandomVariable> parents = new util.ArraySet<RandomVariable>(givens);
		network.connect(forVar, parents, cpt);
	}

	/**
	 * Reads numeric values from the given string, and saves them as the
	 * probability values of entries in the given CPT.
	 * <p>
	 * XMLBIF spec: ``The body of the TABLE tag is a sequence of non-negative
	 * real numbers, in the counting order of the declared variables, taking
	 * the GIVEN variables first and then the FOR variables.''
	 * <p>
	 * Example: P(A|B,C) where A has 3 values, B has 2 values and C has 4 values
	 * <pre>
		p(A1|B1 C1) 
		p(A2|B1 C1) 
		p(A3|B1 C1) 
		p(A1|B1 C2) 
		p(A2|B1 C2) 
		p(A3|B1 C2) 
		p(A1|B1 C3) 
		p(A2|B1 C3) 
		p(A3|B1 C3) 
		p(A1|B1 C4) 
		p(A2|B1 C4) 
		p(A3|B1 C4) 
		p(A1|B2 C1) 
		p(A2|B2 C1) 
		p(A3|B2 C1) 
		p(A1|B2 C2) 
		p(A2|B2 C2) 
		p(A3|B2 C2) 
		p(A1|B2 C3) 
		p(A2|B2 C3) 
		p(A3|B2 C3) 
		p(A1|B2 C4) 
		p(A2|B2 C4) 
		p(A3|B2 C4) 
	 * </pre>
	 */
	public void initCPTFromString(CPT cpt, List<RandomVariable> givens, String str) throws NumberFormatException, CPTFormatException {
		//trace("  initCPTFromString: " + str);
		StringTokenizer tokens = new StringTokenizer(str);
		recursivelyInitCPT(cpt, new bn.base.Assignment(), givens, tokens);
	}
	
	/**
	 * Recursively initialize the given CPT as follows:
	 * <ol>
	 *   <li>If the list of givens (parents) is empty:
	 *     <ul>
	 *       <li>Then use the given Assignment (of values to parents) for the
	 *           ``row'' of the CPT</li>
	 *       <li>For each value of the CPT's variable, read a token, parse it
	 *           as a double, and set that as the probability for the value
	 *           in the row</li>
	 *       <li>Note that this will handle the priors for variables with no
	 *           parents since the first call to this method passes an empty
	 *           Assignment.</li>
	 *     </ul>
	 *   </li>
	 *   <li>Otherwise:
	 *     <ul>
	 *      <li>Remove the first variable from the list of givens (parents)</li>
	 *      <li>For each value of that variable, add the variable=value
	 *          assignment to the Assignment and recurse on the rest of the
	 *          givens.</li>
	 *     </ul>
	 *   </li>
	 * </ol>
	 */
	protected void recursivelyInitCPT(CPT cpt, Assignment a, List<RandomVariable> givens, StringTokenizer tokens) {
		if (givens.isEmpty()) {
			// No givens: Set probabilities for given Assignment 
			for (Value v : cpt.getVariable().getRange()) {
				String token = tokens.nextToken();
				//trace("    recursivelyInitCPT: token=" + token);
				double p = Double.parseDouble(token);
				Assignment aa = a.copy();
				//trace("    recursivelyInitCPT: aa=" + aa);
				cpt.set(v, aa, p);
			}
		} else {
			// Otherwise: iterate over values of first given and recurse on rest
			RandomVariable firstGiven = givens.get(0);
			//trace("    recursivelyInitCPT: firstGiven=" + firstGiven);
			List<RandomVariable> restGivens = givens.subList(1, givens.size());
			for (Value v : firstGiven.getRange()) {
				//trace("    recursivelyInitCPT: " + firstGiven + "=" + variable);
				a.put(firstGiven, v);
				recursivelyInitCPT(cpt, a, restGivens, tokens);
				a.remove(firstGiven);
			}
		}
	}

	protected Element getChildWithTagName(Element elt, String tagname) {
		NodeList children = elt.getChildNodes();
		if (children != null && children.getLength() > 0) {
			for (int i=0; i < children.getLength(); i++) {
				Node node = children.item(i);
				if (node.getNodeType() == Node.ELEMENT_NODE) {
					Element childElt = (Element)node;
					if (childElt.getTagName().equals(tagname)) {
						return childElt;
					}
				}
			}
		}
		throw new NoSuchElementException(tagname);
	}

	protected void doForEachChild(Element elt, String tagname, ElementTaker taker) {
		NodeList children = elt.getChildNodes();
		if (children != null && children.getLength() > 0) {
			for (int i=0; i < children.getLength(); i++) {
				Node node = children.item(i);
				if (node.getNodeType() == Node.ELEMENT_NODE) {
					Element childElt = (Element)node;
					if (childElt.getTagName().equals(tagname)) {
						taker.element(childElt);
					}
				}
			}
		}
	}

	/**
	 * Returns the concatenated child text of the specified node.
	 * This method only looks at the immediate children of type
	 * Node.TEXT_NODE or the children of any child node that is of
	 * type Node.CDATA_SECTION_NODE for the concatenation.
	 */
	public String getChildText(Node node) {
		if (node == null) {
			return null;
		}
		StringBuilder buf = new StringBuilder();
		Node child = node.getFirstChild();
		while (child != null) {
			short type = child.getNodeType();
			if (type == Node.TEXT_NODE) {
				buf.append(child.getNodeValue());
			}
			else if (type == Node.CDATA_SECTION_NODE) {
				buf.append(getChildText(child));
			}
			child = child.getNextSibling();
		}
		return buf.toString();
	}

	protected void trace(String msg) {
		System.err.println(msg);
	}

	/**
	 * Parse an XMLBIF file and print out the resulting BayesianNetwork.
	 * <p>
	 * Usage: java bn.parser.XMLBIFParser FILE
	 * <p>
	 * With no arguments: reads aima-alarm.xml in the src tree 
	 */
	public static void main(String[] argv) throws IOException, ParserConfigurationException, SAXException {
		String filename = "src/bn/examples/aima-alarm.xml";
		if (argv.length > 0) {
			filename = argv[0];
		}
		XMLBIFParser parser = new XMLBIFParser();
		BayesianNetwork network = parser.readNetworkFromFile(filename);
		System.out.println(network);
	}

}

interface ElementTaker {
	public void element(Element e);
}
