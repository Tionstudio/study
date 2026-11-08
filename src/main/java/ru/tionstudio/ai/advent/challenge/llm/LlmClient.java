package ru.tionstudio.ai.advent.challenge.llm;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.function.Consumer;
import java.util.logging.Logger;

/**
 * Client for an OpenAI-compatible chat completions API
 * (OpenAI, OpenRouter, and any other provider exposing
 * {@code POST {baseUrl}/chat/completions}).
 *
 * <p>Uses streaming ({@code "stream": true}): reply tokens are forwarded to
 * a consumer as they arrive, which keeps long generations observable and the
 * connection to the caller alive.</p>
 */
public class LlmClient {

    /** A single chat message (role: system | user | assistant). */
    public record Message(String role, String content) {
    }

    /**
     * Per-request generation options chosen in the web UI settings frame.
     *
     * @param format            response format id: none, json, xml, yaml,
     *                          markdown-table, chain-of-thought, styled
     * @param formatDescription optional free-text description (format for json/xml,
     *                          style for styled)
     * @param maxTokens         max_tokens value or null (natural stop)
     * @param stopSequences     custom stop sequences (may be empty)
     * @param reasoningEffort   reasoning depth for thinking models: low | high | max,
     *                          or null to use the provider default
     * @param temperature       sampling temperature (0..2) or null to not send it
     *                          (only applicable to models that support temperature)
     */
    public record ChatOptions(String format,
                              String formatDescription,
                              Integer maxTokens,
                              List<String> stopSequences,
                              String reasoningEffort,
                              Double temperature) {

        public static ChatOptions defaults() {
            return new ChatOptions("none", "", null, List.of(), null, null);
        }
    }

    /**
     * Result of a single LLM call: full reply text plus token usage
     * reported by the provider in the {@code usage} field (nullable fields —
     * some providers omit or partially fill them).
     *
     * @param finishReason    why generation stopped ("stop", "length", …) or null
     * @param reasoningTokens tokens a reasoning model spent "thinking", or null
     * @param reasoning       full text of the model's hidden "thinking" (may be empty)
     */
    public record ChatResult(String reply,
                             Integer promptTokens,
                             Integer completionTokens,
                             Integer totalTokens,
                             String finishReason,
                             Integer reasoningTokens,
                             String reasoning) {
    }

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final Logger LOG = Logger.getLogger(LlmClient.class.getName());

    private final HttpClient httpClient;
    private final String apiKey;
    private final String endpoint;
    private final String model;

    public LlmClient(String apiKey, String baseUrl, String model) {
        this.apiKey = apiKey;
        this.endpoint = baseUrl.replaceAll("/+$", "") + "/chat/completions";
        this.model = model;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(15))
                .build();
    }

    /** The default model this client sends when a request has no override. */
    public String model() {
        return model;
    }

    /**
     * Sends a message list (history + new user message) to the LLM in streaming
     * mode: every content token is passed to {@code tokenConsumer} and every
     * reasoning token (hidden "thinking" of reasoning models) to
     * {@code reasoningConsumer} as soon as it arrives.
     *
     * @return the accumulated reply together with token usage statistics
     * @throws IOException          on network errors or non-2xx API responses
     * @throws InterruptedException if the request is interrupted
     */
    public ChatResult chatStream(List<Message> messagesToSend, ChatOptions options,
                                 Consumer<String> tokenConsumer, Consumer<String> reasoningConsumer)
            throws IOException, InterruptedException {
        ObjectNode payload = MAPPER.createObjectNode();
        payload.put("model", model);
        payload.put("stream", true);
        // Ask the provider to append a usage chunk at the end of the stream
        payload.putObject("stream_options").put("include_usage", true);

        ArrayNode messages = payload.putArray("messages");
        String systemPrompt = buildSystemPrompt(options);
        if (systemPrompt != null) {
            messages.addObject().put("role", "system").put("content", systemPrompt);
        }
        for (Message m : messagesToSend) {
            messages.addObject().put("role", m.role()).put("content", m.content());
        }

        if (options.maxTokens() != null && options.maxTokens() > 0) {
            payload.put("max_tokens", options.maxTokens());
        }
        if (options.stopSequences() != null && !options.stopSequences().isEmpty()) {
            ArrayNode stop = payload.putArray("stop");
            options.stopSequences().forEach(stop::add);
        }
        if (options.reasoningEffort() != null && !options.reasoningEffort().isBlank()) {
            payload.put("reasoning_effort", options.reasoningEffort());
        }
        if (options.temperature() != null) {
            payload.put("temperature", options.temperature());
        }

        String lastMessage = messagesToSend.isEmpty()
                ? ""
                : messagesToSend.get(messagesToSend.size() - 1).content();
        LOG.info(() -> "LLM request (stream): model=" + model
                + ", format=" + options.format()
                + (systemPrompt != null ? " (system prompt: " + systemPrompt.length() + " chars)" : " (no system prompt)")
                + (options.maxTokens() != null ? ", max_tokens=" + options.maxTokens() : "")
                + (options.stopSequences() != null && !options.stopSequences().isEmpty()
                        ? ", stop=" + options.stopSequences() : "")
                + (options.reasoningEffort() != null && !options.reasoningEffort().isBlank()
                        ? ", reasoning_effort=" + options.reasoningEffort() : "")
                + (options.temperature() != null ? ", temperature=" + options.temperature() : "")
                + ", messages=" + messagesToSend.size()
                + ", last=" + abbreviate(lastMessage, 120));

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(endpoint))
                // Force HTTP/1.1: the default HTTP/2 h2c upgrade stalls against
                // some local OpenAI-compatible servers (LM Studio, llama.cpp)
                .version(HttpClient.Version.HTTP_1_1)
                .timeout(Duration.ofSeconds(120))
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer " + apiKey)
                .header("Accept", "text/event-stream")
                .POST(HttpRequest.BodyPublishers.ofString(MAPPER.writeValueAsString(payload)))
                .build();

        long startNanos = System.nanoTime();
        HttpResponse<InputStream> response = null;
        String errorBody = null;
        // Retry transient failures: providers under parallel load (batch mode)
        // often answer 429/5xx — a short backoff usually gets the request through
        final int maxAttempts = 4;
        final long[] backoffMs = { 2000, 5000, 10000 };
        for (int attempt = 1; attempt <= maxAttempts; attempt++) {
            response = httpClient.send(request, HttpResponse.BodyHandlers.ofInputStream());
            int status = response.statusCode();
            if (status / 100 == 2) {
                break;
            }
            try (InputStream in = response.body()) {
                errorBody = new String(in.readAllBytes(), StandardCharsets.UTF_8);
            }
            response = null;
            boolean retryable = status == 429 || status == 500 || status == 502
                    || status == 503 || status == 504;
            if (retryable && attempt < maxAttempts) {
                long delay = backoffMs[attempt - 1];
                LOG.warning("LLM API error: HTTP " + status + " (attempt " + attempt + "/"
                        + maxAttempts + "), retrying in " + delay + " ms: "
                        + abbreviate(errorBody, 300));
                Thread.sleep(delay);
            } else {
                LOG.warning("LLM API error: HTTP " + status + " (attempt " + attempt + "/"
                        + maxAttempts + "), giving up: " + abbreviate(errorBody, 300));
                throw new IOException("LLM API returned HTTP " + status + ": "
                        + abbreviate(errorBody, 300));
            }
        }

        StringBuilder reply = new StringBuilder();
        StringBuilder reasoning = new StringBuilder();
        Integer promptTokens = null, completionTokens = null, totalTokens = null, reasoningTokens = null;
        String finishReason = null;
        int chunks = 0;

        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(response.body(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (!line.startsWith("data:")) {
                    continue; // comments, keep-alives, event lines
                }
                String data = line.substring(5).trim();
                if (data.isEmpty() || data.equals("[DONE]")) {
                    continue;
                }
                JsonNode chunk = MAPPER.readTree(data);

                JsonNode choice = chunk.path("choices").path(0);
                JsonNode delta = choice.path("delta");
                JsonNode content = delta.path("content");
                if (content.isTextual()) {
                    String token = content.asText();
                    reply.append(token);
                    chunks++;
                    tokenConsumer.accept(token);
                }
                // Reasoning models report hidden "thinking" separately from content
                // (field name differs between providers: "reasoning" / "reasoning_content")
                JsonNode thinking = delta.path("reasoning");
                if (!thinking.isTextual()) {
                    thinking = delta.path("reasoning_content");
                }
                if (thinking.isTextual()) {
                    String token = thinking.asText();
                    reasoning.append(token);
                    reasoningConsumer.accept(token);
                }
                JsonNode finish = choice.path("finish_reason");
                if (finish.isTextual()) {
                    finishReason = finish.asText();
                }

                JsonNode usage = chunk.path("usage");
                if (usage.isObject()) {
                    promptTokens = intOrNull(usage.path("prompt_tokens"), promptTokens);
                    completionTokens = intOrNull(usage.path("completion_tokens"), completionTokens);
                    totalTokens = intOrNull(usage.path("total_tokens"), totalTokens);
                    reasoningTokens = intOrNull(
                            usage.path("completion_tokens_details").path("reasoning_tokens"), reasoningTokens);
                }
            }
        }

        long elapsedMs = (System.nanoTime() - startNanos) / 1_000_000;
        ChatResult result = new ChatResult(
                reply.toString(), promptTokens, completionTokens, totalTokens,
                finishReason, reasoningTokens, reasoning.toString());
        LOG.info("LLM response: HTTP 200 in " + elapsedMs + " ms, " + chunks + " stream chunks"
                + ", reply=" + result.reply().length() + " chars"
                + (result.reasoning().isEmpty() ? "" : ", reasoning=" + result.reasoning().length() + " chars")
                + ", finish_reason=" + result.finishReason()
                + ", tokens (prompt/completion/total)="
                + result.promptTokens() + "/" + result.completionTokens() + "/" + result.totalTokens()
                + (result.reasoningTokens() != null ? ", reasoning=" + result.reasoningTokens() : ""));
        LOG.info("LLM reply text:\n" + result.reply());
        if (result.reply().isEmpty() && "length".equals(result.finishReason())) {
            LOG.warning("Empty reply: the model exhausted max_tokens while still reasoning "
                    + "(reasoning tokens consume the same budget). Increase max_tokens or use natural stop.");
        }
        return result;
    }

    /**
     * Non-streaming convenience call: collects the full reply internally.
     * Used for auxiliary requests like context summarization.
     */
    public ChatResult chatOnce(List<Message> messages, ChatOptions options)
            throws IOException, InterruptedException {
        return chatStream(messages, options, token -> { }, reasoningToken -> { });
    }

    private static String abbreviate(String text, int maxLength) {
        if (text == null) {
            return "";
        }
        String oneLine = text.replaceAll("\\s+", " ").trim();
        return oneLine.length() <= maxLength ? oneLine : oneLine.substring(0, maxLength) + "…";
    }

    private static Integer intOrNull(JsonNode node, Integer fallback) {
        if (node.isInt() || node.isLong()) {
            return node.asInt();
        }
        return fallback;
    }

    /**
     * Builds a system prompt implementing the requested response format.
     * Returns null for "none" (no system message is added).
     */
    private static String buildSystemPrompt(ChatOptions options) {
        if (options == null || options.format() == null) {
            return null;
        }
        String description = options.formatDescription() == null ? "" : options.formatDescription().trim();
        String suffix = description.isEmpty()
                ? ""
                : " The response must follow this format description: " + description;

        return switch (options.format()) {
            case "json" -> "You are a helpful assistant. Respond strictly in valid JSON format. "
                    + "Do not add any text, comments or markdown fences outside the JSON structure." + suffix;
            case "xml" -> "You are a helpful assistant. Respond strictly in valid XML format. "
                    + "Do not add any text, comments or markdown fences outside the XML structure." + suffix;
            case "yaml" -> "You are a helpful assistant. Respond strictly in valid YAML format. "
                    + "Do not add any text, comments or markdown fences outside the YAML document.";
            case "markdown-table" -> "You are a helpful assistant. Present your whole answer as a Markdown table "
                    + "(with a header row and a separator row). Do not add any text outside the table.";
            case "chain-of-thought" -> "You are a helpful assistant. Think step by step: show your full reasoning "
                    + "chain explicitly, and only then give the final answer clearly marked as the conclusion.";
            case "styled" -> description.isEmpty()
                    // No style given: keep the former default — a system analyst's style
                    ? "Ты — опытный системный аналитик. Отвечай в деловом, структурированном стиле: "
                            + "выделяй цели, требования, ограничения и риски; используй терминологию системного анализа; "
                            + "формулируй мысли чётко и лаконично."
                    : "You are a helpful assistant. Write your entire answer strictly in the following style: "
                            + description;
            default -> null; // "none" and unknown values: no system prompt
        };
    }
}
