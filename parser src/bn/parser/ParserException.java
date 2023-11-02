/*
 * File: ParserException.java
 * Creator: George Ferguson
 */

package bn.parser;

import java.io.IOException;

/**
 * Exceptions thrown during parsing.
 */
public class ParserException extends IOException {

	public static final long serialVersionUID = 1L;

	public ParserException(String msg) {
		super(msg);
	}

}
