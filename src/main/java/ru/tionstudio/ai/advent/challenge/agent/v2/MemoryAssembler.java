package ru.tionstudio.ai.advent.challenge.agent.v2;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

/** Builds the visible per-layer token breakdown stored with every LLM call. */
public final class MemoryAssembler {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private MemoryAssembler() { }

    public static ObjectNode stats(String system, String longTerm, String working,
                                   String shortTerm, String user, double calibration) {
        ObjectNode root = MAPPER.createObjectNode();
        add(root, "system", system, calibration);
        add(root, "longTerm", longTerm, calibration);
        add(root, "working", working, calibration);
        add(root, "shortTerm", shortTerm, calibration);
        add(root, "user", user, calibration);
        int estimated = 0;
        // The current user message is already part of the serialized short-term block.
        // Keep it as a separate UI metric, but never count it twice in the request total.
        for (String key : new String[]{"system", "longTerm", "working", "shortTerm"}) {
            estimated += root.path(key).path("estTokens").asInt(0);
        }
        root.put("totalEstTokens", estimated);
        root.put("calibration", calibration);
        return root;
    }

    private static void add(ObjectNode root, String name, String text, double calibration) {
        int chars = text == null ? 0 : text.length();
        ObjectNode item = root.putObject(name);
        item.put("chars", chars);
        item.put("estTokens", TokenEstimator.estimate(text, calibration));
    }
}
