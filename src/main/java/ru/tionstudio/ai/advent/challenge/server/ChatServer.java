package ru.tionstudio.ai.advent.challenge.server;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import ru.tionstudio.ai.advent.challenge.agent.AgentSession;
import ru.tionstudio.ai.advent.challenge.agent.AgentTools;
import ru.tionstudio.ai.advent.challenge.agent.AgentDialogStore;
import ru.tionstudio.ai.advent.challenge.config.AppConfig;
import ru.tionstudio.ai.advent.challenge.conversation.ConversationManager;
import ru.tionstudio.ai.advent.challenge.conversation.ConversationStore;
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
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
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
    /** Model group id -> client, in config order. */
    private final Map<String, LlmClient> clientsById;
    /** Group configs (id + model + base-url) in config order, for /api/models. */
    private final List<AppConfig.ModelConfig> modelConfigs;
    private final LlmClient defaultClient;
    private final LlmClient analysisClient;
    private final PresetStore presetStore;
    private final ConversationManager conversationManager;
    /** File/time tools for the agent page, sandboxed to the test/ directory. */
    private final AgentTools agentTools;
    /** Persistent storage of agent dialogs (transcript + LLM history). */
    private final AgentDialogStore agentDialogStore;
    /** Active agent sessions (in-memory; a session lives until "Новый диалог"). */
    private final Map<String, AgentSession> agentSessions = new ConcurrentHashMap<>();
    /** Pending file-operation permission requests: request id -> decision. */
    private final Map<String, CompletableFuture<Boolean>> permissionRegistry = new ConcurrentHashMap<>();

    public ChatServer(int port, Map<String, LlmClient> clientsById,
                      List<AppConfig.ModelConfig> modelConfigs,
                      LlmClient defaultClient, LlmClient analysisClient,
                      PresetStore presetStore, ConversationManager conversationManager,
                      AgentTools agentTools, AgentDialogStore agentDialogStore) {
        this.port = port;
        this.clientsById = clientsById;
        this.modelConfigs = modelConfigs;
        this.defaultClient = defaultClient;
        this.analysisClient = analysisClient;
        this.presetStore = presetStore;
        this.conversationManager = conversationManager;
        this.agentTools = agentTools;
        this.agentDialogStore = agentDialogStore;
    }

    public void start() throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);
        server.createContext("/api/chat", this::handleChat);
        server.createContext("/api/agent/chat", this::handleAgentChat);
        server.createContext("/api/agent/permission", this::handleAgentPermission);
        server.createContext("/api/agent/control", this::handleAgentControl);
        server.createContext("/api/agent/session", this::handleAgentSession);
        server.createContext("/api/agent/history", this::handleAgentHistory);
        server.createContext("/api/presets", this::handlePresets);
        server.createContext("/api/conversations", this::handleConversations);
        server.createContext("/api/models", this::handleModels);
        server.createContext("/", this::handleStatic);
        server.setExecutor(Executors.newVirtualThreadPerTaskExecutor());
        server.start();
        LOG.info("Server started: http://localhost:" + port + "/");
    }

    /** Finds a client by group id or by model name (case-insensitive). */
    private LlmClient resolveClient(String ref) {
        for (AppConfig.ModelConfig c : modelConfigs) {
            if (c.id().equalsIgnoreCase(ref) || c.model().equalsIgnoreCase(ref)) {
                return clientsById.get(c.id());
            }
        }
        return null;
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
            ConversationManager.ContextSettings contextSettings =
                    ConversationManager.ContextSettings.parse(json);

            // Client selection:
            //  - useCloud=true  → the analysis model (llm.analysis.model), used
            //    by "Анализ результирующей таблицы";
            //  - "model" field  → the chosen model group (id or model name),
            //    e.g. from the chat settings or batch JSON;
            //  - otherwise      → the default model (llm.default-model).
            LlmClient client = defaultClient;
            if (json.path("useCloud").asBoolean(false)) {
                client = analysisClient;
            } else {
                String modelRef = json.path("model").asText(null);
                if (modelRef != null && !modelRef.isBlank()) {
                    LlmClient selected = resolveClient(modelRef.trim());
                    if (selected == null) {
                        sendJson(exchange, 400, errorJson("Unknown model: '" + modelRef
                                + "'. Configure it in config.properties (llm.models)."));
                        return;
                    }
                    client = selected;
                }
            }
            final String effectiveModel = client.model();
            LOG.info("POST /api/chat: model=" + effectiveModel
                    + ", message=" + abbreviate(message, 120));

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
                    // Context mode: register the message, fit history into the
                    // configured limit, maybe compress the overflow (the
                    // summarization LLM call happens here, under the heartbeat).
                    ConversationManager.Prepared prepared = conversationManager.prepare(
                            json.path("conversationId").asText(null), message, contextSettings);
                    if (prepared.conversationId() != null) {
                        ObjectNode convJson = MAPPER.createObjectNode();
                        convJson.put("id", prepared.conversationId());
                        synchronized (out) {
                            writeSse(out, "conversation", convJson.toString());
                        }
                        if (prepared.compressed()) {
                            ObjectNode ctxJson = MAPPER.createObjectNode();
                            ctxJson.put("keptMessages", prepared.keptMessages());
                            ctxJson.put("totalMessages", prepared.totalMessages());
                            ctxJson.put("summary", prepared.summaryUsed());
                            synchronized (out) {
                                writeSse(out, "context", ctxJson.toString());
                            }
                        }
                    }

                    LlmClient.ChatResult result;
                    long startNanos = System.nanoTime();
                    try {
                        result = client.chatStream(prepared.messages(), options, token -> {
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
                        String errText = e.getMessage() != null
                                ? e.getMessage()
                                : e.getClass().getSimpleName() + " (LLM server unreachable?)";
                        synchronized (out) {
                            writeSse(out, "error", errorJson("LLM request failed: " + errText));
                        }
                        return;
                    }
                    long elapsedMs = (System.nanoTime() - startNanos) / 1_000_000;

                    // Context mode: store the assistant's reply in the history
                    conversationManager.onReply(prepared.conversationId(), result.reply());

                    ObjectNode doneJson = MAPPER.createObjectNode();
                    doneJson.put("reply", result.reply());
                    doneJson.set("applied", appliedJson(options, effectiveModel));
                    ObjectNode statsNode = statsJson(result, elapsedMs);
                    if (prepared.auxTokens() > 0) {
                        // Tokens spent on the auxiliary summarization call
                        statsNode.put("auxTokens", prepared.auxTokens());
                    }
                    doneJson.set("stats", statsNode);
                    if (prepared.conversationId() != null) {
                        ObjectNode ctxJson = MAPPER.createObjectNode();
                        ctxJson.put("keptMessages", prepared.keptMessages());
                        ctxJson.put("totalMessages", prepared.totalMessages());
                        ctxJson.put("summaryChars", prepared.summaryChars());
                        ctxJson.put("fillPercent", prepared.fillPercent());
                        doneJson.set("context", ctxJson);
                    }

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
     * Creates (or restores) the agent session with the given id. When a
     * stored dialog with this id exists, its LLM history is replayed into the
     * fresh session so the dialog can be continued after a restart. Either
     * way the session is bound to the persistent transcript.
     */
    private AgentSession createAgentSession(String sid) {
        AgentSession session = new AgentSession(sid, defaultClient, agentTools, permissionRegistry);
        session.setTranscriptListener(new AgentSession.TranscriptListener() {
            @Override
            public void onEntry(String kind, ObjectNode payload) {
                agentDialogStore.appendEntry(sid, kind, payload.toString());
            }

            @Override
            public void onHistory(String role, String content) {
                ObjectNode json = MAPPER.createObjectNode();
                json.put("role", role);
                json.put("content", content);
                agentDialogStore.appendEntry(sid, "history", json.toString());
            }
        });
        if (agentDialogStore.dialogExists(sid)) {
            List<AgentSession.HistoryEntry> restored = new ArrayList<>();
            for (AgentDialogStore.StoredEntry e : agentDialogStore.loadEntries(sid, "history")) {
                try {
                    JsonNode json = MAPPER.readTree(e.payload());
                    restored.add(new AgentSession.HistoryEntry(
                            json.path("role").asText("user"),
                            json.path("content").asText("")));
                } catch (Exception ex) {
                    LOG.warning("Skipping malformed history entry in dialog " + sid);
                }
            }
            if (!restored.isEmpty()) {
                session.restoreHistory(restored);
            }
        }
        return session;
    }

    /**
     * {@code POST /api/agent/chat} — runs one turn of the tool-using agent.
     * Body: {@code {"message": "...", "sessionId"?: "...", "compression"?:
     * "none"|"lastN", "contextMessages"?: N, "maxToolCalls"?: N}}.
     * SSE events: session, token, tool_call, tool_result, permission_request,
     * permission_resolved, paused, resumed, limit, done, error.
     */
    private void handleAgentChat(HttpExchange exchange) throws IOException {
        try {
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                sendJson(exchange, 405, errorJson("Method not allowed. Use POST."));
                return;
            }
            String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            JsonNode json = MAPPER.readTree(body);
            String message = json.path("message").asText("").trim();
            if (message.isEmpty()) {
                sendJson(exchange, 400, errorJson("Field 'message' must not be empty."));
                return;
            }

            String sessionId = json.path("sessionId").asText("").trim();
            boolean newDialog = sessionId.isEmpty();
            if (newDialog) {
                sessionId = UUID.randomUUID().toString();
            }
            final String sid = sessionId;
            // The dialog row (and the "current dialog" pointer) must exist
            // before the session starts persisting its transcript
            if (newDialog || !agentDialogStore.dialogExists(sid)) {
                agentDialogStore.createDialog(sid, abbreviate(message, 60));
            } else {
                agentDialogStore.setCurrentDialog(sid);
            }
            AgentSession session = agentSessions.computeIfAbsent(sid, this::createAgentSession);
            LOG.info("POST /api/agent/chat: session=" + sid
                    + ", message=" + abbreviate(message, 120));
            session.updateSettings(new AgentSession.Settings(
                    json.path("compression").asText("none"),
                    json.path("contextMessages").asInt(0),
                    json.path("maxToolCalls").asInt(0),
                    json.path("planning").asBoolean(true)));

            exchange.getResponseHeaders().set("Content-Type", "text/event-stream; charset=utf-8");
            exchange.getResponseHeaders().set("Cache-Control", "no-cache");
            exchange.sendResponseHeaders(200, 0); // chunked transfer

            try (OutputStream out = exchange.getResponseBody()) {
                ScheduledExecutorService heartbeat = Executors.newSingleThreadScheduledExecutor();
                heartbeat.scheduleAtFixedRate(() -> {
                    try {
                        synchronized (out) {
                            out.write(HEARTBEAT_FRAME);
                            out.flush();
                        }
                    } catch (IOException e) {
                        LOG.warning("Agent client disconnected while waiting — closing stream");
                        exchange.close();
                    }
                }, HEARTBEAT_INTERVAL_SECONDS, HEARTBEAT_INTERVAL_SECONDS, TimeUnit.SECONDS);
                try {
                    AgentSession.Listener listener = (event, data) -> {
                        synchronized (out) {
                            writeSse(out, event, data.toString());
                        }
                    };
                    ObjectNode sessionJson = MAPPER.createObjectNode();
                    sessionJson.put("id", sid);
                    listener.on("session", sessionJson);
                    session.runTurn(message, listener);
                    LOG.info("Agent turn completed, session " + sid);
                } finally {
                    heartbeat.shutdownNow();
                }
            }
        } catch (JsonProcessingException e) {
            LOG.warning("Rejected /api/agent/chat request: malformed JSON body");
            sendQuietly(exchange, 400, errorJson("Invalid JSON body."));
        } catch (UncheckedIOException | IOException e) {
            LOG.warning("Agent client disconnected during streaming: " + e.getMessage());
        } catch (Exception e) {
            LOG.log(Level.SEVERE, "Agent request failed", e);
            sendQuietly(exchange, 500, errorJson("Agent request failed: " + e.getMessage()));
        }
    }

    /**
     * {@code POST /api/agent/permission} — resolves a pending file-operation
     * permission request: {@code {"requestId": "...", "allow": true|false}}.
     */
    private void handleAgentPermission(HttpExchange exchange) throws IOException {
        try {
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                sendJson(exchange, 405, errorJson("Method not allowed. Use POST."));
                return;
            }
            String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            JsonNode json = MAPPER.readTree(body);
            String requestId = json.path("requestId").asText("").trim();
            if (requestId.isEmpty()) {
                sendJson(exchange, 400, errorJson("Field 'requestId' must not be empty."));
                return;
            }
            CompletableFuture<Boolean> decision = permissionRegistry.get(requestId);
            if (decision == null) {
                sendJson(exchange, 404, errorJson(
                        "Permission request not found or already resolved: " + requestId));
                return;
            }
            boolean allow = json.path("allow").asBoolean(false);
            decision.complete(allow);
            LOG.info("Agent permission " + requestId + " resolved: " + (allow ? "allowed" : "denied"));
            sendJson(exchange, 200, "{\"resolved\":true,\"allowed\":" + allow + "}");
        } catch (Exception e) {
            LOG.log(Level.SEVERE, "Agent permission request failed", e);
            sendQuietly(exchange, 500, errorJson("Permission request failed: " + e.getMessage()));
        }
    }

    /**
     * {@code POST /api/agent/control} — pause/resume/abort a running agent
     * session: {@code {"sessionId": "...", "action": "pause"|"resume"|"abort"}}.
     */
    private void handleAgentControl(HttpExchange exchange) throws IOException {
        try {
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                sendJson(exchange, 405, errorJson("Method not allowed. Use POST."));
                return;
            }
            String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            JsonNode json = MAPPER.readTree(body);
            String sessionId = json.path("sessionId").asText("").trim();
            String action = json.path("action").asText("").trim();
            AgentSession session = agentSessions.get(sessionId);
            if (session == null) {
                sendJson(exchange, 404, errorJson("Agent session not found: " + sessionId));
                return;
            }
            switch (action) {
                case "pause" -> session.pause();
                case "resume" -> session.resume();
                case "abort" -> session.abort();
                default -> {
                    sendJson(exchange, 400, errorJson(
                            "Unknown action: '" + action + "'. Use pause | resume | abort."));
                    return;
                }
            }
            LOG.info("Agent session " + sessionId + ": control action " + action);
            sendJson(exchange, 200, "{\"ok\":true,\"action\":\"" + action + "\"}");
        } catch (Exception e) {
            LOG.log(Level.SEVERE, "Agent control request failed", e);
            sendQuietly(exchange, 500, errorJson("Control request failed: " + e.getMessage()));
        }
    }

    /**
     * {@code DELETE /api/agent/session?id=...} — drops an agent session
     * ("Новый диалог" on the agent page).
     */
    private void handleAgentSession(HttpExchange exchange) throws IOException {
        try {
            if (!"DELETE".equalsIgnoreCase(exchange.getRequestMethod())) {
                sendJson(exchange, 405, errorJson("Method not allowed. Use DELETE."));
                return;
            }
            String id = null;
            String rawQuery = exchange.getRequestURI().getRawQuery();
            if (rawQuery != null) {
                for (String pair : rawQuery.split("&")) {
                    int eq = pair.indexOf('=');
                    if (eq > 0 && pair.substring(0, eq).equals("id")) {
                        id = java.net.URLDecoder.decode(pair.substring(eq + 1), StandardCharsets.UTF_8);
                    }
                }
            }
            if (id == null || id.isBlank()) {
                sendJson(exchange, 400, errorJson("Query parameter 'id' is required."));
                return;
            }
            AgentSession removed = agentSessions.remove(id);
            if (removed != null) {
                removed.abort();
            }
            // "Новый диалог": forget the pointer so the app does not restore
            // this dialog after a restart. The stored transcript is kept for
            // a future multi-dialog list.
            agentDialogStore.clearCurrentDialog();
            if (removed != null) {
                LOG.info("DELETE /api/agent/session: " + id + " deleted");
                sendJson(exchange, 200, "{\"deleted\":true}");
            } else {
                LOG.info("DELETE /api/agent/session: " + id + " not found");
                sendJson(exchange, 200, "{\"deleted\":true}");
            }
        } catch (Exception e) {
            LOG.log(Level.SEVERE, "Agent session request failed", e);
            sendQuietly(exchange, 500, errorJson("Session request failed: " + e.getMessage()));
        }
    }

    /**
     * {@code GET /api/agent/history} — returns the current (most recent)
     * agent dialog so the page can re-render it after an application restart:
     * {@code {"sessionId": "...", "entries": [{kind, payload}, ...]}}.
     * The in-memory session is restored as well, so the dialog can simply be
     * continued. Without a current dialog: {@code {"sessionId": null,
     * "entries": []}}.
     */
    private void handleAgentHistory(HttpExchange exchange) throws IOException {
        try {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                sendJson(exchange, 405, errorJson("Method not allowed. Use GET."));
                return;
            }
            ObjectNode response = MAPPER.createObjectNode();
            ArrayNode entries = response.putArray("entries");
            String dialogId = agentDialogStore.currentDialogId();
            if (dialogId != null && agentDialogStore.dialogExists(dialogId)) {
                // Make sure the session is back in memory with its LLM history
                agentSessions.computeIfAbsent(dialogId, this::createAgentSession);
                response.put("sessionId", dialogId);
                for (AgentDialogStore.StoredEntry e
                        : agentDialogStore.loadEntriesExcept(dialogId, "history")) {
                    ObjectNode entry = MAPPER.createObjectNode();
                    entry.put("kind", e.kind());
                    entry.set("payload", MAPPER.readTree(e.payload()));
                    entries.add(entry);
                }
                LOG.info("GET /api/agent/history: dialog " + dialogId
                        + ", " + entries.size() + " entries");
            } else {
                response.putNull("sessionId");
            }
            sendJson(exchange, 200, response.toString());
        } catch (Exception e) {
            LOG.log(Level.SEVERE, "Agent history request failed", e);
            sendQuietly(exchange, 500, errorJson("History request failed: " + e.getMessage()));
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
                case "GET" -> {
                    LOG.info("GET /api/presets");
                    sendJson(exchange, 200, presetStore.listJson());
                }
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
                    boolean deleted = presetStore.delete(id);
                    LOG.info("DELETE /api/presets: id=" + id
                            + (deleted ? " deleted" : " not found"));
                    if (deleted) {
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

    /**
     * {@code GET /api/conversations?id=...} — returns the stored conversation
     * so the page can restore the chat after a reload.
     * {@code DELETE /api/conversations?id=...} — removes the conversation
     * ("Новый диалог" button).
     */
    private void handleConversations(HttpExchange exchange) throws IOException {
        try {
            String id = null;
            String rawQuery = exchange.getRequestURI().getRawQuery();
            if (rawQuery != null) {
                for (String pair : rawQuery.split("&")) {
                    int eq = pair.indexOf('=');
                    if (eq > 0 && pair.substring(0, eq).equals("id")) {
                        id = java.net.URLDecoder.decode(pair.substring(eq + 1), StandardCharsets.UTF_8);
                    }
                }
            }
            if (id == null || id.isBlank()) {
                sendJson(exchange, 400, errorJson("Query parameter 'id' is required."));
                return;
            }
            switch (exchange.getRequestMethod().toUpperCase()) {
                case "GET" -> {
                    LOG.info("GET /api/conversations: id=" + id);
                    ConversationStore.Conversation conv = conversationManager.load(id);
                    if (conv == null) {
                        sendJson(exchange, 404, errorJson("Conversation not found: id=" + id));
                        return;
                    }
                    ObjectNode root = MAPPER.createObjectNode();
                    root.put("id", conv.id());
                    root.put("hasSummary", !conv.summary().isBlank());
                    var arr = root.putArray("messages");
                    for (ConversationStore.StoredMessage m : conv.messages()) {
                        arr.addObject().put("role", m.role()).put("content", m.content());
                    }
                    sendJson(exchange, 200, root.toString());
                }
                case "DELETE" -> {
                    boolean deleted = conversationManager.delete(id);
                    LOG.info("DELETE /api/conversations: id=" + id
                            + (deleted ? " deleted" : " not found"));
                    if (deleted) {
                        sendJson(exchange, 200, "{\"deleted\":true}");
                    } else {
                        sendJson(exchange, 404, errorJson("Conversation not found: id=" + id));
                    }
                }
                default -> sendJson(exchange, 405, errorJson("Method not allowed."));
            }
        } catch (Exception e) {
            LOG.log(Level.SEVERE, "Conversations request failed", e);
            sendQuietly(exchange, 500, errorJson("Conversations request failed: " + e.getMessage()));
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

        // Sampling temperature: sent only for models that support it
        Double temperature = null;
        JsonNode temperatureNode = json.path("temperature");
        if (temperatureNode.isNumber()) {
            double t = temperatureNode.asDouble();
            if (t >= 0 && t <= 2) {
                temperature = t;
            }
        }

        return new LlmClient.ChatOptions(format, formatDescription, maxTokens, stopSequences,
                reasoningEffort, temperature);
    }

    /**
     * GET /api/models — model groups the UI may offer for selection:
     * group id and model name for each group, the default group id
     * and the model used for batch-result analysis.
     */
    private void handleModels(HttpExchange exchange) throws IOException {
        if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
            sendJson(exchange, 405, errorJson("Method not allowed. Use GET."));
            return;
        }
        LOG.fine("GET /api/models");
        ObjectNode root = MAPPER.createObjectNode();
        String defaultId = null;
        for (AppConfig.ModelConfig c : modelConfigs) {
            if (clientsById.get(c.id()) == defaultClient) {
                defaultId = c.id();
            }
        }
        root.put("default", defaultId != null ? defaultId : modelConfigs.get(0).id());
        root.put("defaultModel", defaultClient.model());
        root.put("analysisModel", analysisClient.model());
        ArrayNode arr = root.putArray("models");
        for (AppConfig.ModelConfig c : modelConfigs) {
            ObjectNode item = arr.addObject();
            item.put("id", c.id());
            item.put("model", c.model());
            if (c.pricePrompt() != null) {
                item.put("pricePrompt", c.pricePrompt());
            }
            if (c.priceCompletion() != null) {
                item.put("priceCompletion", c.priceCompletion());
            }
        }
        sendJson(exchange, 200, root.toString());
    }

    /**
     * Serializes the actually applied options so the web page can display
     * them next to each LLM answer.
     */
    private static ObjectNode appliedJson(LlmClient.ChatOptions options, String effectiveModel) {
        ObjectNode applied = MAPPER.createObjectNode();
        applied.put("format", options.format());
        applied.put("model", effectiveModel);
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
        if (options.temperature() != null) {
            applied.put("temperature", options.temperature());
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
