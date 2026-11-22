package ru.tionstudio.ai.advent.challenge.agent.v2;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.Locale;

/** Deterministic invariant checks. Semantic checks are gated separately by the session code. */
public final class InvariantGuard {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private InvariantGuard() { }

    public static ObjectNode checkDeterministic(JsonNode invariants, String candidate,
                                                String action, String fileName) {
        ObjectNode result = MAPPER.createObjectNode();
        ArrayNode violations = result.putArray("violations");
        String normalizedCandidate = candidate == null ? "" : candidate.toLowerCase(Locale.ROOT);
        String normalizedFile = fileName == null ? "" : fileName.toLowerCase(Locale.ROOT);
        boolean fileMutation = "write_file".equals(action) || "append_file".equals(action);
        for (JsonNode invariant : invariants) {
            if (!invariant.path("enabled").asBoolean(true)
                    || !"deterministic".equals(invariant.path("checkType").asText("semantic"))) {
                continue;
            }
            JsonNode config = invariant.path("config");
            for (JsonNode term : config.path("forbiddenTerms")) {
                String value = term.asText("").toLowerCase(Locale.ROOT);
                if (!value.isBlank() && normalizedCandidate.contains(value)) {
                    addViolation(violations, invariant, "Обнаружен запрещённый термин: " + term.asText());
                }
            }
            if (fileMutation && config.path("allowedFileExtensions").isArray()
                    && !config.path("allowedFileExtensions").isEmpty()) {
                boolean allowed = false;
                for (JsonNode extension : config.path("allowedFileExtensions")) {
                    if (normalizedFile.endsWith(extension.asText("").toLowerCase(Locale.ROOT))) allowed = true;
                }
                if (!allowed) {
                    addViolation(violations, invariant, "Недопустимый тип файла: " + normalizedFile);
                }
            }
            if (("delete_file".equals(action) || fileMutation)
                    && matchesExactValue(config.path("protectedFiles"), normalizedFile)) {
                addViolation(violations, invariant, "Защищённый файл нельзя изменять или удалять: " + normalizedFile);
            }
        }
        result.put("allowed", violations.isEmpty());
        result.put("feedback", violations.isEmpty() ? "" : "Детерминированная проверка инвариантов не пройдена.");
        return result;
    }

    private static boolean matchesExactValue(JsonNode values, String candidate) {
        if (!values.isArray()) return false;
        for (JsonNode value : values) {
            if (candidate.equals(value.asText("").toLowerCase(Locale.ROOT))) return true;
        }
        return false;
    }

    private static void addViolation(ArrayNode violations, JsonNode invariant, String evidence) {
        ObjectNode violation = violations.addObject();
        violation.put("invariantId", invariant.path("id").asText());
        violation.put("invariant", invariant.path("text").asText());
        violation.put("evidence", evidence);
    }
}
