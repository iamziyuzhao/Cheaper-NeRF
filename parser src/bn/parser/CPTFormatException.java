/*
 * File: CPTFormatException.java
 * Creator: George Ferguson
 */

package bn.parser;

/**
 * Class of exceptions thrown while parsing CPTs from the {@code table}
 * element of a {@code definition} element in an XMLBIF file.
 */
public class CPTFormatException extends IllegalArgumentException {

    public static final long serialVersionUID = 1L;

    public CPTFormatException() {
	super();
    }

}
