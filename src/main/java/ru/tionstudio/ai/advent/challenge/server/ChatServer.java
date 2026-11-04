package ru.tionstudio.ai.advent.challenge.server;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import ru.tionstudio.ai.advent.challenge.llm.LlmClient;
import ru.tionstudio.ai.advent.challenge.llm.ResponseParser;
import ru.tionstudio.ai.advent.challenge.presets.PresetStore;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Embedded HTTP server:
 * <ul>
 *   <li>{@code POST /api/chat} — accepts {@code {"message": "..."}} JSON,
 *       forwards it to the LLM and returns {@code {"reply": "..."}};</li>
 *   <li>everything else — serves static files from the {@code web/} directory
 *       (the simple JS chat page).</li>
 * </ul>
 */
public class ChatServer {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final Logger LOG = Logger.getLogger(ChatServer.class.getName());
    private static final Path WEB_ROOT = Path.of("web");

    /** Interval between SSE keep-alive comment frames while the model is silent. */
    private static final int HEARTBEAT_INTERVAL_SECONDS = 10;
    /** SSE comment frame — invisible to clients, only keeps the connection busy. */
    private static final byte[] HEARTBEAT_FRAME = ": keep-alive\n\n".getBytes(StandardCharsets.UTF_8);

    private static final Map<String, String> CONTENT_TYPES = Map.of(
            ".html", "text/html; charset=utf-8",
            ".css", "text/css; charset=utf-8",
            ".js", "text/javascript; charset=utf-8",
            ".json", "application/json; charset=utf-8",
            ".png", "image/png",
            ".svg", "image/svg+xml",
            ".ico", "image/x-icon"
    );

    private final int port;
    private final LlmClient llmClient;
    private final PresetStore presetStore;

    public ChatServer(int port, LlmClient llmClient, PresetStore presetStore) {
        this.port = port;
        this.llmClient = llmClient;
        this.presetStore = presetStore;
    }

    public void start() throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);
        server.createContext("/api/chat", this::handleChat);
        server.createContext("/api/presets", this::handlePresets);
        server.createContext("/", this::handleStatic);
        server.setExecutor(Executors.newVirtualThreadPerTaskExecutor());
        server.start();
        LOG.info("Server started: http://localhost:" + port + "/");
    }

    private void handleChat(HttpExchange exchange) throws IOException {
        try {
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                sendJson(exchange, 405, errorJson("Method not allowed. Use POST."));
                return;
            }

            String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            JsonNode json = MAPPER.readTree(body);
            String message = json.path("message").asText("").trim();
            if (message.isEmpty()) {
                LOG.warning("Rejected /api/chat request: empty 'message' field");
                sendJson(exchange, 400, errorJson("Field 'message' must not be empty."));
                return;
            }

            LlmClient.ChatOptions options = parseOptions(json);

            // Server-Sent Events: tokens are pushed to the page as they arrive
            exchange.getResponseHeaders().set("Content-Type", "text/event-stream; charset=utf-8");
            exchange.getResponseHeaders().set("Cache-Control", "no-cache");
            exchange.sendResponseHeaders(200, 0); // chunked transfer

            try (OutputStream out = exchange.getResponseBody()) {
                // Heartbeat: while the model is "thinking" (reasoning models may
                // stay silent for tens of seconds before the first token), send
                // SSE comment frames so the browser connection is never idle —
                // otherwise middleboxes reset it, as happened before.
                ScheduledExecutorService heartbeat = Executors.newSingleThreadScheduledExecutor();
                heartbeat.scheduleAtFixedRate(() -> {
                    try {
                        synchronized (out) {
                            out.write(HEARTBEAT_FRAME);
                            out.flush();
                        }
                    } catch (IOException e) {
                        LOG.warning("Client disconnected while waiting for the model — closing stream");
                        exchange.close();
                    }
                }, HEARTBEAT_INTERVAL_SECONDS, HEARTBEAT_INTERVAL_SECONDS, TimeUnit.SECONDS);

                try {
                    LlmClient.ChatResult result;
                    long startNanos = System.nanoTime();
                    try {
                        result = llmClient.chatStream(message, options, token -> {
                            try {
                                ObjectNode tokenJson = MAPPER.createObjectNode();
                                tokenJson.put("text", token);
                                synchronized (out) {
                                    writeSse(out, "token", tokenJson.toString());
                                }
                            } catch (IOException e) {
                                throw new UncheckedIOException(e);
                            }
                        }, reasoningToken -> {
                            try {
                                ObjectNode tokenJson = MAPPER.createObjectNode();
                                tokenJson.put("text", reasoningToken);
                                synchronized (out) {
                                    writeSse(out, "reasoning", tokenJson.toString());
                                }
                            } catch (IOException e) {
                                throw new UncheckedIOException(e);
                            }
                        });
                    } catch (UncheckedIOException e) {
                        throw e; // client disconnected mid-stream — handled below
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        synchronized (out) {
                            writeSse(out, "error", errorJson("Request was interrupted."));
                        }
                        return;
                    } catch (Exception e) {
                        LOG.log(Level.WARNING, "LLM request failed", e);
                        synchronized (out) {
                            writeSse(out, "error", errorJson("LLM request failed: " + e.getMessage()));
                        }
                        return;
                    }
                    long elapsedMs = (System.nanoTime() - startNanos) / 1_000_000;

                    ObjectNode doneJson = MAPPER.createObjectNode();
                    doneJson.put("reply", result.reply());
                    doneJson.set("applied", appliedJson(options));
                    doneJson.set("stats", statsJson(result, elapsedMs));

                    // Optional reply post-processing ("Парсинг ответа" settings section)
                    String parseMode = json.path("parseMode").asText("none");
                    if (!"none".equals(parseMode)) {
                        String parseExpression = json.path("parseExpression").asText("");
                        doneJson.put("parseMode", parseMode);
                        try {
                            String parsed = ResponseParser.apply(parseMode, parseExpression, result.reply());
                            doneJson.put("parsed", parsed);
                            LOG.info("Parsing (" + parseMode + ") succeeded: expression='" + parseExpression
                                    + "', result=" + abbreviate(parsed, 200));
                        } catch (Exception e) {
                            doneJson.put("parseError", String.valueOf(e.getMessage()));
                            LOG.warning("Parsing (" + parseMode + ") failed: expression='" + parseExpression
                                    + "', error=" + e.getMessage());
                        }
                    }

                    synchronized (out) {
                        writeSse(out, "done", doneJson.toString());
                    }
                    LOG.info("Chat request completed in " + elapsedMs + " ms");
                } finally {
                    heartbeat.shutdownNow();
                }
            }
        } catch (JsonProcessingException e) {
            LOG.warning("Rejected /api/chat request: malformed JSON body");
            sendQuietly(exchange, 400, errorJson("Invalid JSON body."));
        } catch (UncheckedIOException | IOException e) {
            // The client did not wait for the stream to finish — nothing to deliver to.
            LOG.warning("Client disconnected during streaming: " + e.getMessage());
        } catch (Exception e) {
            LOG.log(Level.SEVERE, "Chat request failed", e);
            sendQuietly(exchange, 500, errorJson("LLM request failed: " + e.getMessage()));
        }
    }

    /**
     * Writes one Server-Sent Events frame and flushes it immediately,
     * so the page receives tokens in real time.
     */
    private static void writeSse(OutputStream out, String event, String jsonData) throws IOException {
        out.write(("event: " + event + "\ndata: " + jsonData + "\n\n").getBytes(StandardCharsets.UTF_8));
        out.flush();
    }

    /**
     * {@code GET /api/presets} — lists saved settings presets (defaults + user's).
     * {@code POST /api/presets} — upserts a preset: {"name": "...", "settings": {...}}.
     */
    private void handlePresets(HttpExchange exchange) throws IOException {
        try {
            switch (exchange.getRequestMethod().toUpperCase()) {
                case "GET" -> sendJson(exchange, 200, presetStore.listJson());
                case "POST" -> {
                    String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
                    JsonNode json = MAPPER.readTree(body);
                    String name = json.path("name").asText("").trim();
                    JsonNode settings = json.path("settings");
                    if (name.isEmpty()) {
                        sendJson(exchange, 400, errorJson("Field 'name' must not be empty."));
                        return;
                    }
                    if (!settings.isObject()) {
                        sendJson(exchange, 400, errorJson("Field 'settings' must be an object."));
                        return;
                    }
                    String saved = presetStore.saveJson(name, settings);
                    LOG.info("Preset saved: '" + name + "'");
                    sendJson(exchange, 200, saved);
                }
                case "DELETE" -> {
                    Long id = parseIdParam(exchange.getRequestURI().getRawQuery());
                    if (id == null) {
                        sendJson(exchange, 400, errorJson("Query parameter 'id' must be a number."));
                        return;
                    }
                    if (presetStore.delete(id)) {
                        sendJson(exchange, 200, "{\"deleted\":true,\"id\":" + id + "}");
                    } else {
                        sendJson(exchange, 404, errorJson("Preset not found: id=" + id));
                    }
                }
                default -> sendJson(exchange, 405, errorJson("Method not allowed."));
            }
        } catch (PresetStore.PresetLimitExceededException e) {
            LOG.warning("Preset limit reached: " + e.getMessage());
            sendQuietly(exchange, 409, errorJson(e.getMessage()));
        } catch (Exception e) {
            LOG.log(Level.SEVERE, "Presets request failed", e);
            sendQuietly(exchange, 500, errorJson("Presets request failed: " + e.getMessage()));
        }
    }

    /** Parses the {@code id} query parameter; returns {@code null} if missing or malformed. */
    private static Long parseIdParam(String rawQuery) {
        if (rawQuery == null) {
            return null;
        }
        for (String pair : rawQuery.split("&")) {
            int eq = pair.indexOf('=');
            if (eq > 0 && pair.substring(0, eq).equals("id")) {
                try {
                    return Long.parseLong(pair.substring(eq + 1));
                } catch (NumberFormatException e) {
                    return null;
                }
            }
        }
        return null;
    }

    /**
     * Sends an error response, tolerating an already-dead connection —
     * there is no one left to report the delivery failure to.
     */
    private static void sendQuietly(HttpExchange exchange, int status, String json) {
        try {
            sendJson(exchange, status, json);
        } catch (IOException e) {
            LOG.fine("Could not deliver error response, client disconnected: " + e.getMessage());
        }
    }

    private static String abbreviate(String text, int maxLength) {
        if (text == null) {
            return "";
        }
        String oneLine = text.replaceAll("\\s+", " ").trim();
        return oneLine.length() <= maxLength ? oneLine : oneLine.substring(0, maxLength) + "…";
    }

    /**
     * Parses the settings-frame fields of the request JSON into {@link LlmClient.ChatOptions}.
     * Expected fields (all optional):
     * <ul>
     *   <li>{@code format} — none | json | xml | yaml | markdown-table | chain-of-thought | styled;</li>
     *   <li>{@code formatDescription} — free text, used for json/xml;</li>
     *   <li>{@code maxTokens} — integer, used when stop mode is max_tokens;</li>
     *   <li>{@code stopSequences} — comma-separated string of custom stop sequences;</li>
     *   <li>{@code reasoningEffort} — low | high | max (thinking models only).</li>
     * </ul>
     */
    private static LlmClient.ChatOptions parseOptions(JsonNode json) {
        String format = json.path("format").asText("none");
        String formatDescription = json.path("formatDescription").asText("");

        Integer maxTokens = null;
        JsonNode maxTokensNode = json.path("maxTokens");
        if (maxTokensNode.isInt() && maxTokensNode.asInt() > 0) {
            maxTokens = maxTokensNode.asInt();
        }

        String stopRaw = json.path("stopSequences").asText("");
        List<String> stopSequences = Arrays.stream(stopRaw.split(","))
                .map(String::trim)
                .filter(s -> !s.isEmpty())
                .toList();

        String reasoningEffort = json.path("reasoningEffort").asText(null);

        return new LlmClient.ChatOptions(format, formatDescription, maxTokens, stopSequences, reasoningEffort);
    }

    /**
     * Serializes the actually applied options so the web page can display
     * them next to each LLM answer.
     */
    private static ObjectNode appliedJson(LlmClient.ChatOptions options) {
        ObjectNode applied = MAPPER.createObjectNode();
        applied.put("format", options.format());
        if (options.formatDescription() != null && !options.formatDescription().isBlank()) {
            applied.put("formatDescription", options.formatDescription());
        }
        if (options.maxTokens() != null) {
            applied.put("maxTokens", options.maxTokens());
        }
        if (options.stopSequences() != null && !options.stopSequences().isEmpty()) {
            applied.putPOJO("stopSequences", options.stopSequences());
        }
        if (options.reasoningEffort() != null && !options.reasoningEffort().isBlank()) {
            applied.put("reasoningEffort", options.reasoningEffort());
        }
        return applied;
    }

    /**
     * Serializes call statistics: measured round-trip time and token usage
     * (only the fields the provider actually returned).
     */
    private static ObjectNode statsJson(LlmClient.ChatResult result, long elapsedMs) {
        ObjectNode stats = MAPPER.createObjectNode();
        stats.put("elapsedMs", elapsedMs);
        if (result.promptTokens() != null) {
            stats.put("promptTokens", result.promptTokens());
        }
        if (result.completionTokens() != null) {
            stats.put("completionTokens", result.completionTokens());
        }
        if (result.totalTokens() != null) {
            stats.put("totalTokens", result.totalTokens());
        }
        if (result.finishReason() != null) {
            stats.put("finishReason", result.finishReason());
        }
        if (result.reasoningTokens() != null) {
            stats.put("reasoningTokens", result.reasoningTokens());
        }
        return stats;
    }

    private void handleStatic(HttpExchange exchange) throws IOException {
        String path = exchange.getRequestURI().getPath();
        if (path.equals("/")) {
            path = "/index.html";
        }

        // Guard against path traversal (../../etc)
        Path file = WEB_ROOT.resolve(path.substring(1)).normalize();
        if (!file.startsWith(WEB_ROOT.normalize()) || !Files.isRegularFile(file)) {
            sendText(exchange, 404, "Not found: " + path);
            return;
        }

        String name = file.getFileName().toString();
        String contentType = "application/octet-stream";
        int dot = name.lastIndexOf('.');
        if (dot >= 0) {
            contentType = CONTENT_TYPES.getOrDefault(name.substring(dot), contentType);
        }

        byte[] bytes;
        try (InputStream in = Files.newInputStream(file)) {
            bytes = in.readAllBytes();
        }
        exchange.getResponseHeaders().set("Content-Type", contentType);
        exchange.sendResponseHeaders(200, bytes.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }

    private static String errorJson(String message) {
        ObjectNode node = MAPPER.createObjectNode();
        node.put("error", message);
        return node.toString();
    }

    private static void sendJson(HttpExchange exchange, int status, String json) throws IOException {
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        exchange.sendResponseHeaders(status, bytes.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }

    private static void sendText(HttpExchange exchange, int status, String text) throws IOException {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "text/plain; charset=utf-8");
        exchange.sendResponseHeaders(status, bytes.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }
}
