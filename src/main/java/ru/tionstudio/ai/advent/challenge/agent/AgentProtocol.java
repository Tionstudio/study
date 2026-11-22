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
            "update_plan", "complete_plan_step", "skip_plan_step", "transition_task", "ask_user", "ask_user_choice");

    /** Hard cap on the number of tool calls accepted in a single reply. */
    public static final int MAX_CALLS_PER_REPLY = 5;

    /** A parsed tool call: name plus the raw args object. */
    public record ToolCall(String tool, JsonNode args) {
    }

    /** System prompt pinned at the head of the agent's context (never compressed). */
    public static String systemPrompt() {
        return systemPrompt(true);
    }

    /**
     * System prompt for the agent. With tools disabled the model is told to
     * behave as a plain chat assistant and to never emit tool-call JSON.
     */
    public static String systemPrompt(boolean withTools) {
        String common = """
                Ты — универсальный ассистент с явно переданной памятью диалога.

                Используй контекст следующим образом:
                - LONG-TERM — постоянные правила поведения, стиля, формата и ограничений.
                - WORKING — текущая задача, её состояние и промежуточные результаты.
                - SHORT-TERM — сводка, сохранённые факты и последние сообщения диалога.

                Соблюдай включённые правила LONG-TERM. Явные пожелания пользователя в текущем запросе
                могут уточнять стиль, объём и формат ответа, но не отменяют ограничения безопасности.

                Сохранённые факты считай сведениями из предыдущего диалога, а не независимо проверенными
                данными. При противоречиях ориентируйся на более позднюю явную информацию и при необходимости
                уточни её у пользователя.

                Отвечай на языке пользователя. Не сообщай о внутренних слоях памяти и технических запросах,
                если пользователь не спрашивает о работе агента. Не выдумывай факты, действия или результаты.
                """;
        if (!withTools) {
            return common + """
                    У тебя НЕТ доступа к внешним инструментам, файлам и системному времени.
                    Не вызывай внешние инструменты. Внутренние команды плана разрешены:
                    complete_plan_step — завершить текущий шаг: {"tool":"complete_plan_step","args":{"result":"полученный результат"}}.
                    skip_plan_step — пропустить текущий шаг: {"tool":"skip_plan_step","args":{"reason":"причина"}}.
                    ask_user — задать открытый уточняющий вопрос и остановиться до ответа: {"tool":"ask_user","args":{"question":"вопрос"}}.
                    ask_user_choice — предложить варианты выбора и остановиться: {"tool":"ask_user_choice","args":{"question":"вопрос","options":[{"id":"demo","label":"Сделать демо"}],"allowCustom":true}}.
                    update_plan — изменить состав плана только если он стал неправильным.
                    Выполняй шаги последовательно. Не помечай шаг выполненным без фактического результата.
                    Если для текущего шага нужны сведения пользователя, остановись через ask_user или ask_user_choice. Если можно предложить понятные альтернативы (например, «Указать мои данные» и «Сделать демо»), ОБЯЗАТЕЛЬНО используй ask_user_choice, а не открытый вопрос. Не принимай решение за пользователя и не пропускай такой шаг самостоятельно.
                    Финальный ответ давай обычным текстом только после завершения всех шагов.
                    Если пользователь просит запомнить факт, подтверди это и учитывай факт в дальнейших ответах.
                    """;
        }
        return common + """
                У тебя есть доступ к инструментам на сервере пользователя.

                Доступные инструменты:
                1. get_system_time — текущее системное время сервера. Аргументы: нет.
                2. list_files — список файлов в рабочей директории (имя, размер, дата изменения). Аргументы: нет.
                3. read_file — прочитать файл из рабочей директории. Аргументы: {"name": string}.
                4. write_file — создать или полностью перезаписать файл в рабочей директории. Аргументы: {"name": string, "content": string}. Требует подтверждения пользователя.
                5. append_file — дописать текст в конец файла (файл создаётся, если его нет). Аргументы: {"name": string, "content": string}. Требует подтверждения пользователя.
                6. delete_file — удалить файл из рабочей директории. Аргументы: {"name": string}. Требует подтверждения пользователя.
                7. calculate — вычислить арифметическое выражение. Аргументы: {"expression": string}. Поддерживаются числа, + - * / % ^ и скобки.
                8. update_plan — изменить состав плана, если он стал неправильным. Аргументы: {"steps": [{"title": string, "requiresUserInput": boolean, "requiredForValidation": boolean}, ...]}. Не используй для отметки прогресса или обхода обязательных шагов.
                9. complete_plan_step — завершить ТОЛЬКО текущий шаг после фактического получения результата. Аргументы: {"result": string}. Код активирует следующий шаг.
                10. skip_plan_step — пропустить ТОЛЬКО текущий необязательный шаг. Аргументы: {"reason": string}. Причина обязательна. Обязательный для валидации шаг пропустить нельзя; шаг, требующий ввода пользователя, можно пропустить только после его явного разрешения.
                11. transition_task — запросить переход конечного автомата. Аргументы: {"stage": "research"|"planning"|"execution"|"validation"|"done", "step": string, "expected": string}. Переход применяется только если разрешён кодом. Не пытайся перейти в done: завершение выполняется после отдельной валидации.
                12. ask_user — задать открытый уточняющий вопрос и остановиться до ответа. Аргументы: {"question": string}.
                13. ask_user_choice — предложить варианты выбора и остановиться до ответа. Аргументы: {"question": string, "options": [{"id": string, "label": string}, ...], "allowCustom": boolean}. От 2 до 6 вариантов.
                Команды ask_user и ask_user_choice вызывай отдельно, без других инструментов.

                Выполняй план последовательно: сначала получи результат текущего шага, затем вызови complete_plan_step.
                Если нужен свободный текст, вызови ask_user. Если можно предложить понятные альтернативы (например, «Указать мои данные» и «Сделать демо»), ОБЯЗАТЕЛЬНО используй ask_user_choice, а не открытый вопрос. Не принимай решение за пользователя и не пропускай такой шаг самостоятельно.
                Не объединяй complete_plan_step, skip_plan_step, ask_user или ask_user_choice с другими вызовами в параллельный массив.
                Финальный ответ давай только после завершения или обоснованного пропуска всех шагов.

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
    public static String planningPrompt(String userTask) {
        return planningPrompt(userTask, "");
    }

    public static String planningPrompt(String userTask, String invariants) {
        return """
                Составь краткий план решения задачи пользователя. Ответь СТРОГО одним JSON-объектом, без пояснений и markdown:
                {"steps": [{"title":"краткая формулировка шага", "requiresUserInput":false, "requiredForValidation":true}], "appliedInvariantIds":["id"]}
                От 1 до 7 шагов. Если задача простая — план может состоять из одного шага.
                requiresUserInput=true для любого шага, где нужны данные, решение или подтверждение пользователя
                (например, определить содержание, уточнить требования, выбрать вариант). Не пропускай такие шаги самостоятельно.
                requiredForValidation=true для шагов, без фактического результата которых задача не решена.
                Для необязательных дополнений используй requiredForValidation=false. Хотя бы один шаг должен быть обязательным.
                Если инварианты заданы, перечисли все учтённые идентификаторы в appliedInvariantIds и не планируй их нарушение.

                %s

                Задача пользователя: %s
                """.formatted(invariants, userTask);
    }

    public record PlanStepSpec(String title, boolean requiresUserInput, boolean requiredForValidation) { }

    /** Parses structured plan steps, retaining the legacy string format as a fallback. */
    public static List<PlanStepSpec> parsePlanSteps(String reply) {
        List<PlanStepSpec> steps = new ArrayList<>();
        if (reply != null) {
            String text = reply.trim();
            if (text.startsWith("```")) {
                text = text.replaceFirst("^```(?:json)?\\s*", "");
                int end = text.lastIndexOf("```");
                if (end >= 0) text = text.substring(0, end);
                text = text.trim();
            }
            JsonNode json = tryParseJson(text);
            if (json == null) {
                int start = text.indexOf('{');
                int end = text.lastIndexOf('}');
                if (start >= 0 && end > start) json = tryParseJson(text.substring(start, end + 1));
            }
            if (json != null && json.path("steps").isArray()) {
                for (JsonNode s : json.path("steps")) {
                    if (s.isTextual() && !s.asText().isBlank()) {
                        steps.add(new PlanStepSpec(s.asText().trim(), false, true));
                    } else if (s.isObject() && s.path("title").isTextual()
                            && !s.path("title").asText().isBlank()) {
                        steps.add(new PlanStepSpec(s.path("title").asText().trim(),
                                s.path("requiresUserInput").asBoolean(false),
                                s.path("requiredForValidation").asBoolean(true)));
                    }
                }
            }
            if (steps.isEmpty()) {
                for (String line : text.split("\n")) {
                    String cleaned = line.trim().replaceFirst("^[-*•\\d.)\s]+", "").trim();
                    if (!cleaned.matches("[\"{}\\[\\],:\\s]*") && !cleaned.isEmpty() && !cleaned.startsWith("```")) {
                        steps.add(new PlanStepSpec(cleaned, false, true));
                    }
                }
            }
        }
        if (steps.isEmpty()) steps.add(new PlanStepSpec("Выполнить задачу пользователя", false, true));
        while (steps.size() > 10) steps.remove(steps.size() - 1);
        return steps;
    }

    public static List<String> parsePlan(String reply) {
        return parsePlanSteps(reply).stream().map(PlanStepSpec::title).toList();
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
        // Weak models sometimes drop the opening '{' (and even the opening
        // quote) of the call object: {"tool": ... arrives as tool": ... }
        text = repairBareToolKeys(text);
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
     * Repairs the most common structural damage in weak models' tool calls:
     * a dropped '{' that opens the call object and/or a dropped opening
     * quote of the "tool" key. Inserts the missing pieces before every
     * bare "tool" key; occurrences inside string literals are left alone.
     */
    private static String repairBareToolKeys(String text) {
        StringBuilder out = new StringBuilder(text.length() + 8);
        boolean inString = false, escaped = false;
        int i = 0;
        while (i < text.length()) {
            char c = text.charAt(i);
            if (escaped) {
                out.append(c);
                escaped = false;
                i++;
                continue;
            }
            if (c == '\\' && inString) {
                out.append(c);
                escaped = true;
                i++;
                continue;
            }
            if (!inString) {
                // Quoted key: "tool" followed (after optional spaces) by ':'
                if (c == '"' && text.startsWith("\"tool\"", i)
                        && isKeyColon(text, i + 6)) {
                    insertBraceIfBare(out);
                    out.append(text, i, i + 6);
                    i += 6;
                    continue;
                }
                // Unquoted key: tool" — the opening quote was dropped
                if (c == 't' && text.startsWith("tool\"", i)
                        && isKeyColon(text, i + 5)) {
                    insertBraceIfBare(out);
                    out.append('"').append(text, i, i + 5);
                    i += 5;
                    continue;
                }
            }
            if (c == '"') {
                inString = !inString;
            }
            out.append(c);
            i++;
        }
        return out.toString();
    }

    /** True when the next non-space character at {@code from} is ':'. */
    private static boolean isKeyColon(String text, int from) {
        int j = from;
        while (j < text.length() && Character.isWhitespace(text.charAt(j))) {
            j++;
        }
        return j < text.length() && text.charAt(j) == ':';
    }

    /**
     * Appends '{' when the last non-space character already emitted is not
     * '{' or ',' — i.e. the "tool" key does not open its object.
     */
    private static void insertBraceIfBare(StringBuilder out) {
        int j = out.length() - 1;
        while (j >= 0 && Character.isWhitespace(out.charAt(j))) {
            j--;
        }
        char prev = j >= 0 ? out.charAt(j) : ' ';
        if (prev != '{' && prev != ',') {
            out.append('{');
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
     * Heuristic: the reply was MEANT to be a tool call (it carries a "tool"
     * key, quoted or with a dropped opening quote) but could not be parsed
     * into any valid call. Such replies are reported as protocol errors —
     * they must never be shown to the user as a normal chat answer.
     */
    public static boolean looksLikeToolCall(String reply) {
        if (reply == null) {
            return false;
        }
        String text = reply.trim();
        if (text.startsWith("```")) {
            text = text.replaceFirst("^```(?:json)?\\s*", "");
        }
        return text.contains("\"tool\"") || text.contains("tool\":");
    }

    /**
     * When the model's reply was truncated by the token limit
     * (finish_reason=length), the tail often contains an incomplete JSON
     * tool call or an unclosed markdown fence — showing it raw looks broken.
     * Removes that dangling tail, keeping the prose part.
     */
    public static String stripTruncatedTail(String text) {
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
            case "complete_plan_step" -> {
                if (!args.path("result").isTextual() || args.path("result").asText().isBlank()) {
                    return "Поле 'result' обязательно: кратко укажите фактически полученный результат.";
                }
                return null;
            }
            case "skip_plan_step" -> {
                if (!args.path("reason").isTextual() || args.path("reason").asText().isBlank()) {
                    return "Поле 'reason' обязательно: укажите причину пропуска шага.";
                }
                return null;
            }
            case "transition_task" -> {
                if (!args.path("stage").isTextual() || args.path("stage").asText().isBlank()) {
                    return "Поле 'stage' обязательно и должно быть строкой.";
                }
                return null;
            }
            case "ask_user", "ask_user_choice" -> {
                if (!args.path("question").isTextual() || args.path("question").asText().isBlank()) {
                    return "Поле 'question' обязательно и должно быть непустой строкой.";
                }
                if ("ask_user_choice".equals(tool)) {
                    JsonNode options = args.path("options");
                    if (!options.isArray() || options.size() < 2 || options.size() > 6) {
                        return "Поле 'options' должно содержать от 2 до 6 вариантов.";
                    }
                    for (JsonNode option : options) {
                        if (!option.path("label").isTextual() || option.path("label").asText().isBlank()) {
                            return "У каждого варианта должна быть непустая строка 'label'.";
                        }
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

    // ----- Sticky Facts (key-value memory, compression mode "sticky") -----

    /** One extracted fact: a fixed category plus a free-form key and value. */
    public record Fact(String category, String key, String value) {
    }

    /** Allowed fact categories; anything else is stored as "fact". */
    public static final Set<String> FACT_CATEGORIES =
            Set.of("goal", "constraint", "preference", "decision", "fact");

    /** Hard caps for one extraction result. */
    public static final int MAX_FACTS_PER_EXTRACTION = 20;
    public static final int MAX_FACT_KEY_CHARS = 100;
    public static final int MAX_FACT_VALUE_CHARS = 500;

    /**
     * Prompt for the fact-extraction call. The call is made WITHOUT the dialog
     * context: just this text plus the instruction to answer with strict JSON.
     */
    public static String factExtractionPrompt(String text, boolean fromAssistant) {
        return """
                Ты извлекаешь факты для долговременной памяти агента (ключ-значение).
                Ответь СТРОГО одним JSON-объектом, без пояснений и markdown:
                {"facts": [{"category": "...", "key": "...", "value": "..."}, ...]}
                Допустимые category: goal (цель пользователя), constraint (ограничение),
                preference (предпочтение), decision (решение или договорённость), fact (прочий факт).
                key — короткое имя факта (2-5 слов, по-русски), value — краткая суть.
                Извлекай только явно сказанное, ничего не додумывай.
                Если фактов нет, ответь {"facts": []}.

                %s:
                %s
                """.formatted(fromAssistant
                        ? "Ответ ассистента (извлеки принятые решения и договорённости)"
                        : "Сообщение пользователя", text);
    }

    /**
     * Prompt for the periodic consolidation of the stored fact list
     * (merge duplicates/synonyms, drop contradicted facts).
     */
    public static String factConsolidationPrompt(String factsJson) {
        return """
                Ниже — список фактов долговременной памяти агента в формате JSON.
                Объедини дубли и синонимы (оставь самое свежее значение), удали факты,
                которым новые факты явно противоречат. Ничего не добавляй от себя.
                Ответь СТРОГО одним JSON-объектом тем же форматом, без пояснений:
                {"facts": [{"category": "...", "key": "...", "value": "..."}, ...]}

                Факты:
                %s
                """.formatted(factsJson);
    }

    /** Detects semantic contradictions without allowing the model to mutate storage. */
    public static String factConflictPrompt(String newFactsJson, String storedFactsJson) {
        return """
                Сравни новые факты с сохранёнными. Найди только ЯВНЫЕ противоречия:
                два значения описывают одно свойство или решение и не могут быть актуальны
                одновременно. Разные независимые сведения и дополняющие детали конфликтом не являются.
                Совпадение нормализованного ключа обрабатывается кодом отдельно, его не возвращай.
                Ответь СТРОГО одним JSON-объектом без markdown:
                {"conflicts":[{"newIndex":0,"existingId":123}]}
                newIndex — индекс нового факта, existingId — id единственного заменяемого факта.
                Если противоречий нет, верни {"conflicts":[]}.

                Новые факты: %s
                Сохранённые факты: %s
                """.formatted(newFactsJson, storedFactsJson);
    }

    /**
     * Parses the fact-extraction reply. Tolerates markdown fences and prose
     * around the JSON object; invalid entries are skipped. Never null.
     */
    public static List<Fact> parseFacts(String reply) {
        List<Fact> facts = new ArrayList<>();
        if (reply == null) {
            return facts;
        }
        String text = reply.trim();
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
            JsonNode arr = json.path("facts");
            if (arr.isArray()) {
                for (JsonNode f : arr) {
                    String category = f.path("category").asText("fact").trim().toLowerCase();
                    if (!FACT_CATEGORIES.contains(category)) {
                        category = "fact";
                    }
                    String key = f.path("key").asText("").trim();
                    String value = f.path("value").asText("").trim();
                    if (key.isEmpty() || value.isEmpty()) {
                        continue;
                    }
                    if (key.length() > MAX_FACT_KEY_CHARS) {
                        key = key.substring(0, MAX_FACT_KEY_CHARS);
                    }
                    if (value.length() > MAX_FACT_VALUE_CHARS) {
                        value = value.substring(0, MAX_FACT_VALUE_CHARS);
                    }
                    facts.add(new Fact(category, key, value));
                    if (facts.size() >= MAX_FACTS_PER_EXTRACTION) {
                        break;
                    }
                }
            }
        }
        return facts;
    }
}
