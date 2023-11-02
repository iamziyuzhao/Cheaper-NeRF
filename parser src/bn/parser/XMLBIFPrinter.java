package bn.parser;

import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;

import javax.xml.parsers.ParserConfigurationException;

import org.xml.sax.SAXException;

import bn.core.Assignment;
import bn.core.BayesianNetwork;
import bn.core.Named;
import bn.core.RandomVariable;
import bn.core.Value;

/**
 * Prints BayesianNetworks in XMLBIF format.
 * @see http://sites.poli.usp.br/p/fabio.cozman/Research/InterchangeFormat/index.html
 */
public class XMLBIFPrinter {
	
	public XMLBIFPrinter(PrintStream out) {
		this.out = out;
	}
	
	public String doubleFormatString = "%g"; // default precision is 6
	
	protected PrintStream out;
	
	public void print(BayesianNetwork network) {
		print(network, null);
	}

	public void print(BayesianNetwork network, String name) {
		printXMLHeader();
		printDOCTYPE();
		printBIFHeader();
		printNetwork(network, name);
		printBIFTrailer();
	}
	
	protected void printXMLHeader() {
		out.println("<?xml version=\"1.0\"?>");
	}

	protected void printDOCTYPE() {
		out.println("<!-- DTD for the XMLBIF 0.3 format -->\n" + 
				"<!DOCTYPE BIF [\n" + 
				"	<!ELEMENT BIF ( NETWORK )*>\n" + 
				"	      <!ATTLIST BIF VERSION CDATA #REQUIRED>\n" + 
				"	<!ELEMENT NETWORK ( NAME, ( PROPERTY | VARIABLE | DEFINITION )* )>\n" + 
				"	<!ELEMENT NAME (#PCDATA)>\n" + 
				"	<!ELEMENT VARIABLE ( NAME, ( OUTCOME |  PROPERTY )* ) >\n" + 
				"	      <!ATTLIST VARIABLE TYPE (nature|decision|utility) \"nature\">\n" + 
				"	<!ELEMENT OUTCOME (#PCDATA)>\n" + 
				"	<!ELEMENT DEFINITION ( FOR | GIVEN | TABLE | PROPERTY )* >\n" + 
				"	<!ELEMENT FOR (#PCDATA)>\n" + 
				"	<!ELEMENT GIVEN (#PCDATA)>\n" + 
				"	<!ELEMENT TABLE (#PCDATA)>\n" + 
				"	<!ELEMENT PROPERTY (#PCDATA)>\n" + 
				"]>");
	}

	protected void printBIFHeader() {
		out.println("<BIF VERSION=\"0.3\">");
	}

	protected void printBIFTrailer() {
		out.println("</BIF>");
	}

	protected void printNetwork(BayesianNetwork network, String name) {
		out.println("<NETWORK>");
		if (name != null) {
			out.println("<NAME>" + name + "</NAME>");
		}
		// Variables
		for (RandomVariable var : network.getVariables()) {
			out.println("<VARIABLE TYPE=\"nature\">");
			out.println("  <NAME>" + getNameOrDie(var) + "</NAME>");
			for (Value value : var.getRange()) {
				out.println("  <OUTCOME>" + value + "</OUTCOME>");
			}
			out.println("</VARIABLE>");
		}
		// CPTs
		// This requires access to some of the internals of BayesianNetworks, which
		// I have now made public
		for (RandomVariable var : network.getVariables()) {
			out.println("<DEFINITION>");
			out.println("  <FOR>" + getNameOrDie(var) + "</FOR>");
			// Parents must be ordered and that order is used for printing the TABLE
			List<RandomVariable> givens = new ArrayList<RandomVariable>(network.getParents(var)); 
			for (RandomVariable given : givens) {
				out.println("  <GIVEN>" + getNameOrDie(given) + "</GIVEN>");
			}
			out.println("  <TABLE>");
			recursivelyPrintTable(network, new bn.base.Assignment(), var, givens);
			out.println("  </TABLE>");
			out.println("</DEFINITION>");
		}
		out.println("</NETWORK>");
	}
	
	/**
	 * If the given RandomVariable is Named, return its name, otherwise throw
	 * an IllegalArgumentException (which is an unchecked exception FYI).
	 */
	protected String getNameOrDie(RandomVariable var) throws IllegalArgumentException {
		if (var instanceof Named) {
			Named nvar = (Named)var;
			return nvar.getName();
		} else {
			throw new IllegalArgumentException("unnamed RandomVariable cannot be expressed using XMBIF: " + var);
		}
	}
	
	/**
	 * XMLBIF spec: ``The body of the TABLE tag is a sequence of non-negative
	 * real numbers, in the counting order of the declared variables, taking
	 * the GIVEN variables first and then the FOR variables.''
	 * @see XMLBIFParser.initCPTFromString
	 * @see XMLBIFParser.recursivelyInitCPT
	 */
	protected void recursivelyPrintTable(BayesianNetwork network, Assignment a, RandomVariable forVar, List<RandomVariable> givens) {
		if (givens.isEmpty()) {
			// No givens: print the probabilities for the values of forVar with the given assignment
			out.print("    ");
			for (Value v : forVar.getRange()) {
				a.put(forVar, v);
				double p = network.getProbability(forVar, a);
				out.format(doubleFormatString + " ", p);
				a.remove(forVar);
			}
			out.println();
		} else {
			// Otherwise: iterate over values of first given and recurse on rest
			RandomVariable firstGiven = givens.get(0);
			List<RandomVariable> restGivens = givens.subList(1, givens.size());
			for (Value v : firstGiven.getRange()) {
				a.put(firstGiven, v);
				recursivelyPrintTable(network, a, forVar, restGivens);
				a.remove(firstGiven);
			}
		}
	}

	/**
	 * Parse an XMLBIF file and print out the resulting BayesianNetwork as XMLBIF.
	 * <p>
	 * Usage: java bn.parser.XMLBIFPrinter FILE [NETWORK-NAME]
	 * <p>
	 * With no arguments: reads aima-alarm.xml in the src tree 
	 */
	public static void main(String[] argv) throws IOException, ParserConfigurationException, SAXException {
		String filename = "src/bn/examples/aima-alarm.xml";
		String networkName = "AIMA-Alarm";
		if (argv.length > 0) {
			filename = argv[0];
			if (argv.length > 1) {
				networkName = argv[1];
			}
		}
		XMLBIFParser xp = new XMLBIFParser();
		XMLBIFPrinter printer = new XMLBIFPrinter(System.out);
		BayesianNetwork network = xp.readNetworkFromFile(filename);
		printer.print(network, networkName);
    }

}
