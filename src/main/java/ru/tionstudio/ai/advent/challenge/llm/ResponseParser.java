package ru.tionstudio.ai.advent.challenge.llm;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jayway.jsonpath.JsonPath;
import org.w3c.dom.Document;
import org.xml.sax.InputSource;

import javax.xml.XMLConstants;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.xpath.XPath;
import javax.xml.xpath.XPathFactory;
import java.io.StringReader;

/**
 * Post-processing of the raw LLM reply text with a query language
 * chosen in the web UI ("Парсинг ответа" settings section):
 * <ul>
 *   <li>{@code jsonpath} — Jayway JsonPath against the reply parsed as JSON;</li>
 *   <li>{@code xpath} — XPath against the reply parsed as XML.</li>
 * </ul>
 * Any failure (malformed reply text, invalid expression, no match)
 * is reported as an exception — the caller shows it in the chat.
 */
public class ResponseParser {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private ResponseParser() {
    }

    /**
     * Applies the parsing expression to the reply text.
     *
     * @return extracted value as text (objects/arrays serialized to JSON)
     * @throws Exception if parsing fails for any reason
     */
    public static String apply(String mode, String expression, String text) throws Exception {
        if (expression == null || expression.isBlank()) {
            throw new IllegalArgumentException("Не задано выражение для парсинга.");
        }
        return switch (mode) {
            case "jsonpath" -> applyJsonPath(expression.trim(), text);
            case "xpath" -> applyXPath(expression.trim(), text);
            default -> throw new IllegalArgumentException("Неизвестный режим парсинга: " + mode);
        };
    }

    private static String applyJsonPath(String expression, String text) throws Exception {
        Object result = JsonPath.read(text, expression);
        if (result == null) {
            throw new IllegalStateException("JSONPath вернул null (ничего не найдено).");
        }
        return result instanceof String s ? s : MAPPER.writeValueAsString(result);
    }

    private static String applyXPath(String expression, String text) throws Exception {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        // XXE hardening: no DTDs, no external entities
        factory.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true);
        factory.setFeature(XMLConstants.FEATURE_SECURE_PROCESSING, true);
        factory.setExpandEntityReferences(false);

        Document doc = factory.newDocumentBuilder().parse(new InputSource(new StringReader(text)));
        XPath xpath = XPathFactory.newInstance().newXPath();
        String result = xpath.evaluate(expression, doc);
        if (result == null || result.isEmpty()) {
            throw new IllegalStateException("XPath вернул пустой результат (ничего не найдено).");
        }
        return result;
    }
}
