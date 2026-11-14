package ru.tionstudio.ai.advent.challenge.agent;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.json.JsonMapper;
import com.fasterxml.jackson.core.json.JsonReadFeature;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Text protocol between the agent and the LLM.
 *
 * <p>The system prompt lists the available tools and the call format: to use
 * tools the model answers with either a single JSON object
 * {@code {"tool": "<name>", "args": {...}}} or — for several independent
 * calls at once — a JSON array of such objects. Any other reply is treated
 * as the final answer to the user.</p>
 *
 * <p>{@link #parse} extracts the tool calls from the model's reply,
 * {@link #validate} checks each call <em>before</em> any tool is executed —
 * unknown tool names, wrong argument types and oversized values are rejected
 * and reported back to the model as a tool error.</p>
 */
public class AgentProtocol {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    /**
     * Tolerant reader for model output: weak local models often emit raw
     * newlines/tabs inside JSON strings, single quotes or trailing commas —
     * all technically invalid JSON that is still perfectly understandable.
     */
    private static final ObjectMapper LENIENT = JsonMapper.builder()
            .enable(JsonReadFeature.ALLOW_UNESCAPED_CONTROL_CHARS)
            .enable(JsonReadFeature.ALLOW_SINGLE_QUOTES)
            .enable(JsonReadFeature.ALLOW_TRAILING_COMMA)
            .build();

    private static final Set<String> TOOLS = Set.of(
            "get_system_time", "list_files", "read_file",
            "write_file", "append_file", "delete_file", "calculate",
            "update_plan");

    /** Hard cap on the number of tool calls accepted in a single reply. */
    static final int MAX_CALLS_PER_REPLY = 5;

    /** A parsed tool call: name plus the raw args object. */
    public record ToolCall(String tool, JsonNode args) {
    }

    /** System prompt pinned at the head of the agent's context (never compressed). */
    public static String systemPrompt() {
        return """
                Ты — агент-помощник с доступом к инструментам на сервере пользователя.

                Доступные инструменты:
                1. get_system_time — текущее системное время сервера. Аргументы: нет.
                2. list_files — список файлов в рабочей директории (имя, размер, дата изменения). Аргументы: нет.
                3. read_file — прочитать файл из рабочей директории. Аргументы: {"name": string}.
                4. write_file — создать или полностью перезаписать файл в рабочей директории. Аргументы: {"name": string, "content": string}. Требует подтверждения пользователя.
                5. append_file — дописать текст в конец файла (файл создаётся, если его нет). Аргументы: {"name": string, "content": string}. Требует подтверждения пользователя.
                6. delete_file — удалить файл из рабочей директории. Аргументы: {"name": string}. Требует подтверждения пользователя.
                7. calculate — вычислить арифметическое выражение. Аргументы: {"expression": string}. Поддерживаются числа, + - * / % ^ и скобки.
                8. update_plan — обновить план выполнения задачи. Аргументы: {"steps": [{"title": string, "status": "pending"|"in_progress"|"done"}, ...]}. Вызывай при переходе к следующему шагу плана, чтобы пользователь видел прогресс.

                Правила вызова:
                - Чтобы вызвать инструмент, ответь СТРОГО одним JSON-объектом без пояснений, без markdown и без любого другого текста:
                  {"tool": "<имя инструмента>", "args": {...}}
                  Если аргументов нет, укажи "args": {}.
                - Если нужно вызвать НЕСКОЛЬКО независимых инструментов сразу (например, прочитать два файла), ответь JSON-МАССИВОМ (не более 5 вызовов):
                  [{"tool": "<имя>", "args": {...}}, {"tool": "<имя>", "args": {...}}]
                  Независимые вызовы выполняются параллельно.
                - После вызова ты получишь результат(ы) отдельными сообщениями. Проанализируй их и продолжи: вызови следующие инструменты или дай финальный ответ.
                - Финальный ответ пользователю давай обычным текстом (НЕ JSON).
                - Если пользователь запретил операцию или инструмент вернул ошибку, не повторяй её без изменений — объясни ситуацию или предложи другой путь.
                """;
    }

    /**
     * Prompt for the planning phase: the model returns the plan as strict JSON.
     */
    static String planningPrompt(String userTask) {
        return """
                Составь краткий план решения задачи пользователя. Ответь СТРОГО одним JSON-объектом, без пояснений и markdown:
                {"steps": ["краткая формулировка шага 1", "шаг 2", ...]}
                От 1 до 7 шагов. Если задача простая — план может состоять из одного шага.

                Задача пользователя: %s
                """.formatted(userTask);
    }

    /**
     * Parses the planning-phase reply into step titles (never null, at least
     * one step; falls back to numbered/plain lines when the JSON is broken).
     */
    static List<String> parsePlan(String reply) {
        List<String> steps = new ArrayList<>();
        if (reply != null) {
            String text = reply.trim();
            // Strip ```json fences the same way as for tool calls
            if (text.startsWith("```")) {
                text = text.replaceFirst("^```(?:json)?\\s*", "");
                int end = text.lastIndexOf("```");
                if (end >= 0) {
                    text = text.substring(0, end);
                }
                text = text.trim();
            }
            JsonNode json = tryParseJson(text);
            if (json == null) {
                // Prose around the object: try the outermost {...} span
                int start = text.indexOf('{');
                int end = text.lastIndexOf('}');
                if (start >= 0 && end > start) {
                    json = tryParseJson(text.substring(start, end + 1));
                }
            }
            if (json != null) {
                JsonNode arr = json.path("steps");
                if (arr.isArray()) {
                    for (JsonNode s : arr) {
                        if (s.isTextual() && !s.asText().isBlank()) {
                            steps.add(s.asText().trim());
                        }
                    }
                }
            }
            if (steps.isEmpty()) {
                // Fallback: numbered or bulleted lines, one step per line;
                // skip lines that are pure JSON punctuation
                for (String line : text.split("\n")) {
                    String cleaned = line.trim()
                            .replaceFirst("^[-*•\\d.)\s]+", "").trim();
                    if (cleaned.matches("[\"{}\\[\\],:\\s]*")) {
                        continue;
                    }
                    if (!cleaned.isEmpty() && !cleaned.startsWith("```")) {
                        steps.add(cleaned);
                    }
                }
            }
        }
        if (steps.isEmpty()) {
            steps.add("Выполнить задачу пользователя");
        }
        while (steps.size() > 10) {
            steps.remove(steps.size() - 1);
        }
        return steps;
    }

    private static JsonNode tryParseJson(String text) {
        try {
            return LENIENT.readTree(text);
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * Full parse outcome: the tool calls plus the prose the model wrote
     * around them (empty when the reply was pure JSON). The prose is shown
     * to the user instead of the raw JSON bubble.
     */
    public record ParseResult(List<ToolCall> calls, String prose) {
    }

    /**
     * Parses the model reply as tool calls. Returns an empty list when the
     * reply is not a JSON tool call (single object or array) — i.e. it is a
     * regular final answer. At most {@link #MAX_CALLS_PER_REPLY} calls are
     * accepted; extra ones are silently dropped.
     */
    public static List<ToolCall> parse(String reply) {
        return parseFull(reply).calls();
    }

    /**
     * Same as {@link #parse}, but also returns any prose surrounding the
     * extracted tool call(s) so the UI can keep showing it.
     */
    public static ParseResult parseFull(String reply) {
        List<ToolCall> calls = new ArrayList<>();
        if (reply == null) {
            return new ParseResult(calls, "");
        }
        String text = reply.trim();
        if (text.isEmpty()) {
            return new ParseResult(calls, "");
        }
        // Tolerate ```json fences around the payload
        if (text.startsWith("```")) {
            text = text.replaceFirst("^```(?:json)?\\s*", "");
            int end = text.lastIndexOf("```");
            if (end >= 0) {
                text = text.substring(0, end);
            }
            text = text.trim();
        }
        boolean looksJson = (text.startsWith("{") && text.endsWith("}"))
                || (text.startsWith("[") && text.endsWith("]"));
        if (looksJson) {
            JsonNode json = tryParseJson(text);
            if (json != null) {
                collectCalls(json, calls);
                if (!calls.isEmpty()) {
                    return new ParseResult(calls, "");
                }
            }
            // broken JSON or not a tool call — fall through to embedded extraction
        }
        // Weak models often wrap calls in prose ("Сейчас вызову...").
        // Fall back to extracting balanced {...} objects with a "tool" field.
        String prose = extractEmbeddedCalls(text, calls);
        return new ParseResult(calls, prose);
    }

    /** Collects tool calls from a parsed JSON node (object or array). */
    private static void collectCalls(JsonNode json, List<ToolCall> calls) {
        if (json.isArray()) {
            for (JsonNode item : json) {
                if (calls.size() >= MAX_CALLS_PER_REPLY) {
                    return;
                }
                collectCalls(item, calls);
            }
            return;
        }
        JsonNode tool = json.path("tool");
        if (tool.isTextual()) {
            JsonNode args = json.path("args");
            calls.add(new ToolCall(tool.asText(),
                    args.isObject() ? args : MAPPER.createObjectNode()));
        }
    }

    /**
     * Fallback for {@link #parseFull}: scans the reply for balanced JSON
     * objects with a textual "tool" field. Returns the surrounding prose
     * (text before the first and after the last extracted call, trimmed) —
     * or an empty string when nothing was found.
     */
    private static String extractEmbeddedCalls(String text, List<ToolCall> calls) {
        int firstStart = -1;
        int lastEnd = -1;
        for (int start = text.indexOf('{'); start >= 0 && calls.size() < MAX_CALLS_PER_REPLY;
                start = text.indexOf('{', start + 1)) {
            int depth = 0;
            boolean inString = false, escaped = false;
            for (int i = start; i < text.length(); i++) {
                char c = text.charAt(i);
                if (escaped) {
                    escaped = false;
                    continue;
                }
                if (c == '\\' && inString) {
                    escaped = true;
                } else if (c == '"') {
                    inString = !inString;
                } else if (!inString) {
                    if (c == '{') {
                        depth++;
                    } else if (c == '}') {
                        depth--;
                        if (depth == 0) {
                            JsonNode json = tryParseJson(text.substring(start, i + 1));
                            JsonNode tool = json == null ? null : json.path("tool");
                            if (tool != null && tool.isTextual()) {
                                JsonNode args = json.path("args");
                                calls.add(new ToolCall(tool.asText(),
                                        args.isObject() ? args : MAPPER.createObjectNode()));
                                if (firstStart < 0) {
                                    firstStart = start;
                                }
                                lastEnd = i + 1;
                            }
                            break;
                        }
                    }
                }
            }
        }
        if (calls.isEmpty()) {
            return "";
        }
        // If the calls came from a JSON array, swallow its brackets too so
        // they do not leak into the prose ("позвоню [ ]")
        int pStart = firstStart;
        while (pStart > 0 && Character.isWhitespace(text.charAt(pStart - 1))) {
            pStart--;
        }
        if (pStart > 0 && text.charAt(pStart - 1) == '[') {
            pStart--;
        }
        int pEnd = lastEnd;
        while (pEnd < text.length() && Character.isWhitespace(text.charAt(pEnd))) {
            pEnd++;
        }
        if (pEnd < text.length() && text.charAt(pEnd) == ']') {
            pEnd++;
        }
        String prose = (text.substring(0, pStart) + " " + text.substring(pEnd)).trim();
        // Drop leftover markdown fence markers from the prose
        prose = prose.replace("```json", "").replace("```", "").trim();
        return prose;
    }

    /**
     * When the model's reply was truncated by the token limit
     * (finish_reason=length), the tail often contains an incomplete JSON
     * tool call or an unclosed markdown fence — showing it raw looks broken.
     * Removes that dangling tail, keeping the prose part.
     */
    static String stripTruncatedTail(String text) {
        String result = text;
        // Unclosed ``` fence at the tail
        int fences = 0;
        for (int i = result.indexOf("```"); i >= 0; i = result.indexOf("```", i + 3)) {
            fences++;
        }
        if (fences % 2 == 1) {
            result = result.substring(0, result.lastIndexOf("```"));
        }
        // Unbalanced '{' tail (incomplete JSON object)
        int depth = 0;
        int unclosedStart = -1;
        boolean inString = false, escaped = false;
        for (int i = 0; i < result.length(); i++) {
            char c = result.charAt(i);
            if (escaped) {
                escaped = false;
                continue;
            }
            if (c == '\\' && inString) {
                escaped = true;
            } else if (c == '"') {
                inString = !inString;
            } else if (!inString) {
                if (c == '{') {
                    if (depth == 0) {
                        unclosedStart = i;
                    }
                    depth++;
                } else if (c == '}') {
                    depth--;
                    if (depth == 0) {
                        unclosedStart = -1;
                    }
                }
            }
        }
        if (depth > 0 && unclosedStart >= 0) {
            result = result.substring(0, unclosedStart);
        }
        return result.stripTrailing();
    }

    /**
     * Validates a tool call before execution. Returns {@code null} when the
     * call is valid, otherwise a human-readable error (fed back to the model
     * as the tool result).
     */
    public static String validate(ToolCall call) {
        String tool = call.tool();
        if (!TOOLS.contains(tool)) {
            return "Неизвестный инструмент '" + tool + "'. Доступные: " + TOOLS + ".";
        }
        JsonNode args = call.args();
        switch (tool) {
            case "get_system_time", "list_files" -> {
                return null;
            }
            case "read_file", "delete_file" -> {
                return checkName(args);
            }
            case "write_file", "append_file" -> {
                String err = checkName(args);
                if (err != null) {
                    return err;
                }
                JsonNode content = args.path("content");
                if (!content.isTextual()) {
                    return "Поле 'content' обязательно и должно быть строкой.";
                }
                if (content.asText().length() > 100_000) {
                    return "Слишком большое содержимое (максимум 100000 символов).";
                }
                return null;
            }
            case "calculate" -> {
                JsonNode expr = args.path("expression");
                if (!expr.isTextual() || expr.asText().isBlank()) {
                    return "Поле 'expression' обязательно и должно быть непустой строкой.";
                }
                if (expr.asText().length() > 1000) {
                    return "Слишком длинное выражение (максимум 1000 символов).";
                }
                if (!expr.asText().matches("[0-9+\\-*/%^().\\s]+")) {
                    return "В выражении допустимы только цифры, + - * / % ^ , скобки и пробелы.";
                }
                return null;
            }
            case "update_plan" -> {
                JsonNode steps = args.path("steps");
                if (!steps.isArray() || steps.isEmpty()) {
                    return "Поле 'steps' обязательно и должно быть непустым массивом "
                            + "[{\"title\": string, \"status\": \"pending|in_progress|done\"}].";
                }
                for (JsonNode s : steps) {
                    if (!s.path("title").isTextual() || s.path("title").asText().isBlank()) {
                        return "У каждого шага плана должно быть непустое поле 'title'.";
                    }
                    String status = s.path("status").asText("pending");
                    if (!Set.of("pending", "in_progress", "done").contains(status)) {
                        return "Недопустимый статус шага '" + status
                                + "'. Допустимые: pending, in_progress, done.";
                    }
                }
                return null;
            }
            default -> {
                return "Неизвестный инструмент '" + tool + "'.";
            }
        }
    }

    private static String checkName(JsonNode args) {
        JsonNode name = args.path("name");
        if (!name.isTextual() || name.asText().isBlank()) {
            return "Поле 'name' обязательно и должно быть непустой строкой.";
        }
        if (name.asText().length() > 255) {
            return "Слишком длинное имя файла (максимум 255 символов).";
        }
        return null;
    }
}
