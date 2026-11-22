package ru.tionstudio.ai.advent.challenge.server;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import ru.tionstudio.ai.advent.challenge.agent.AgentProtocol;
import ru.tionstudio.ai.advent.challenge.agent.AgentTools;
import ru.tionstudio.ai.advent.challenge.agent.v2.AgentV2Session;
import ru.tionstudio.ai.advent.challenge.agent.v2.MemoryStore;
import ru.tionstudio.ai.advent.challenge.config.AppConfig;
import ru.tionstudio.ai.advent.challenge.llm.LlmClient;

import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

/** HTTP contour of the v2 memory-layer agent. */
public class AgentV2Controller {
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final Logger LOG = Logger.getLogger(AgentV2Controller.class.getName());
    private static final byte[] HEARTBEAT = ": keep-alive\n\n".getBytes(StandardCharsets.UTF_8);

    private final Map<String, LlmClient> clientsById;
    private final List<AppConfig.ModelConfig> modelConfigs;
    private final LlmClient defaultClient;
    private final AgentTools tools;
    private final MemoryStore store;
    private final Map<String, CompletableFuture<Boolean>> permissionRegistry;
    private final Map<String, AgentV2Session> sessions = new ConcurrentHashMap<>();

    public AgentV2Controller(Map<String, LlmClient> clientsById,
                             List<AppConfig.ModelConfig> modelConfigs,
                             LlmClient defaultClient, AgentTools tools,
                             Map<String, CompletableFuture<Boolean>> permissionRegistry) {
        this.clientsById = clientsById;
        this.modelConfigs = modelConfigs;
        this.defaultClient = defaultClient;
        this.tools = tools;
        this.permissionRegistry = permissionRegistry;
        this.store = new MemoryStore(Path.of("settings.db"));
    }

    public void register(HttpServer server) {
        server.createContext("/api/v2/agent/chat", this::handleChat);
        server.createContext("/api/v2/agent/control", this::handleControl);
        server.createContext("/api/v2/agent/history", this::handleHistory);
        server.createContext("/api/v2/agent/dialog/new", this::handleNewDialog);
        server.createContext("/api/v2/agent/dialog/profile", this::handleDialogProfile);
        server.createContext("/api/v2/agent/profiles", this::handleProfiles);
        server.createContext("/api/v2/agent/invariants", this::handleInvariants);
        server.createContext("/api/v2/agent/memory", this::handleMemory);
        server.createContext("/api/v2/agent/rule-candidate", this::handleRuleCandidate);
        server.createContext("/api/v2/agent/reset", this::handleReset);
        server.createContext("/api/v2/agent/llmlog", this::handleLlmLog);
        server.createContext("/api/v2/agent/actionlog", this::handleActionLog);
        server.createContext("/api/v2/agent/action", this::handleAction);
        server.createContext("/api/v2/agent/analyze", this::handleAnalyze);
    }

    private void handleChat(HttpExchange exchange) throws IOException {
        String sid = null;
        try {
            if (!requireMethod(exchange, "POST")) return;
            JsonNode json = readJson(exchange);
            String message = json.path("message").asText("").trim();
            if (message.isEmpty()) {
                sendJson(exchange, 400, errorJson("Field 'message' must not be empty."));
                return;
            }
            LlmClient client = resolveClient(json.path("model").asText(""));
            if (client == null) {
                sendJson(exchange, 400, errorJson("Unknown main model."));
                return;
            }
            sid = json.path("sessionId").asText("").trim();
            if (sid.isEmpty()) sid = UUID.randomUUID().toString();
            if (!store.dialogExists(sid)) {
                store.createDialog(sid, abbreviate(message, 60), json.path("profileId").asText(""));
            } else {
                store.setCurrentDialog(sid);
            }
            AgentV2Session running = sessions.get(sid);
            if (running != null && !running.model().equals(client.model())) {
                running.abort();
                sessions.remove(sid, running);
            }
            AgentV2Session session = sessionFor(sid, client);
            session.updateSettings(new AgentV2Session.Settings(
                    json.path("compression").asText("summary"),
                    json.path("contextMessages").asInt(20),
                    json.path("maxToolCalls").asInt(5),
                    json.path("planning").asBoolean(false),
                    json.path("summaryKeep").asInt(5),
                    json.path("summaryEvery").asInt(10),
                    json.path("slidingMessages").asInt(30),
                    json.path("stickyKeep").asInt(20),
                    json.path("factsMax").asInt(50),
                    json.path("factsConsolidate").asInt(0),
                    json.path("tools").asBoolean(false),
                    json.path("stickyFacts").asBoolean(true)));
            ObjectNode actionDetails = MAPPER.createObjectNode();
            actionDetails.put("model", client.model());
            actionDetails.put("message", abbreviate(message, 200));
            store.logAction(sid, "message_sent", actionDetails.toString());

            exchange.getResponseHeaders().set("Content-Type", "text/event-stream; charset=utf-8");
            exchange.getResponseHeaders().set("Cache-Control", "no-cache");
            exchange.sendResponseHeaders(200, 0);
            try (OutputStream out = exchange.getResponseBody()) {
                ScheduledExecutorService heartbeat = Executors.newSingleThreadScheduledExecutor();
                heartbeat.scheduleAtFixedRate(() -> {
                    try {
                        synchronized (out) {
                            out.write(HEARTBEAT);
                            out.flush();
                        }
                    } catch (IOException e) {
                        exchange.close();
                    }
                }, 10, 10, TimeUnit.SECONDS);
                try {
                    AgentV2Session.Listener listener = (event, data) -> {
                        synchronized (out) {
                            writeSse(out, event, data.toString());
                        }
                    };
                    ObjectNode sessionJson = MAPPER.createObjectNode();
                    sessionJson.put("id", sid);
                    sessionJson.put("title", store.dialogTitle(sid));
                    listener.on("session", sessionJson);
                    session.runTurn(message, listener);
                } finally {
                    heartbeat.shutdownNow();
                }
            }
        } catch (UncheckedIOException | IOException e) {
            LOG.warning("Agent v2 client disconnected: " + e.getMessage());
        } catch (Exception e) {
            LOG.log(Level.SEVERE, "Agent v2 chat failed", e);
            sendQuietly(exchange, 500, errorJson("Agent v2 request failed: " + e.getMessage()));
        }
    }

    private AgentV2Session sessionFor(String sid, LlmClient client) {
        AgentV2Session existing = sessions.get(sid);
        if (existing != null) {
            // Read-only history/memory requests must not replace the selected main model.
            return existing;
        }
        AgentV2Session session = new AgentV2Session(sid, client, tools, store, permissionRegistry);
        session.setContextWindow(contextWindow(client));
        session.setTranscriptListener(new AgentV2Session.TranscriptListener() {
            @Override
            public void onEntry(String kind, ObjectNode payload) {
                store.appendEntry(sid, kind, payload.toString());
            }

            @Override
            public void onHistory(String role, String content) {
                ObjectNode item = MAPPER.createObjectNode();
                item.put("role", role);
                item.put("content", content);
                store.appendEntry(sid, "history", item.toString());
            }
        });
        List<AgentV2Session.HistoryEntry> history = new ArrayList<>();
        for (MemoryStore.StoredEntry e : store.loadEntries(sid, "history")) {
            try {
                JsonNode item = MAPPER.readTree(e.payload());
                history.add(new AgentV2Session.HistoryEntry(
                        item.path("role").asText("user"), item.path("content").asText("")));
            } catch (Exception ignored) {
                // Skip only a malformed stored entry, not the whole dialog.
            }
        }
        session.restoreHistory(history);
        session.restoreFacts(store.listFacts(sid));
        restoreLatestMetadata(session, sid);
        sessions.put(sid, session);
        return session;
    }

    private void restoreLatestMetadata(AgentV2Session session, String sid) {
        for (MemoryStore.StoredEntry e : store.loadEntriesExcept(sid, "history")) {
            try {
                JsonNode payload = MAPPER.readTree(e.payload());
                if ("summary".equals(e.kind())) {
                    session.restoreSummary(payload.path("text").asText(""),
                            payload.path("coveredCounted").asInt(0));
                } else if ("usage".equals(e.kind())) {
                    session.restoreUsage(payload.path("promptTokens").asInt(0),
                            payload.path("completionTokens").asInt(0));
                }
            } catch (Exception ignored) {
                // Metadata is best-effort; transcript/history remain usable.
            }
        }
    }

    private void handleControl(HttpExchange exchange) throws IOException {
        try {
            if (!requireMethod(exchange, "POST")) return;
            JsonNode json = readJson(exchange);
            String sid = json.path("sessionId").asText("");
            String action = json.path("action").asText("");
            AgentV2Session session = sessions.get(sid);
            if (session == null) {
                sendJson(exchange, 404, errorJson("Agent v2 session not found."));
                return;
            }
            switch (action) {
                case "pause" -> session.pause();
                case "resume" -> session.resume();
                case "abort" -> session.abort();
                default -> {
                    sendJson(exchange, 400, errorJson("Use pause | resume | abort."));
                    return;
                }
            }
            store.logAction(sid, action, "{}");
            ObjectNode response = MAPPER.createObjectNode();
            response.put("ok", true);
            response.put("action", action);
            sendJson(exchange, 200, response.toString());
        } catch (Exception e) {
            LOG.log(Level.WARNING, "Agent v2 control failed", e);
            sendQuietly(exchange, 500, errorJson(e.getMessage()));
        }
    }

    private void handleHistory(HttpExchange exchange) throws IOException {
        try {
            if (!requireMethod(exchange, "GET")) return;
            String sid = queryParam(exchange, "dialog");
            if (sid == null || sid.isBlank()) sid = store.currentDialogId();
            ObjectNode response = MAPPER.createObjectNode();
            ArrayNode entries = response.putArray("entries");
            if (sid != null && store.dialogExists(sid)) {
                AgentV2Session session = sessionFor(sid, defaultClient);
                response.put("sessionId", sid);
                response.put("title", store.dialogTitle(sid));
                for (MemoryStore.StoredEntry e : store.loadEntriesExcept(sid, "history")) {
                    if ("turnstats".equals(e.kind())) {
                        response.set("lastTurn", MAPPER.readTree(e.payload()));
                    } else if (Set.of("summary", "facts", "settings", "working", "usage").contains(e.kind())) {
                        continue;
                    }
                    ObjectNode item = entries.addObject();
                    item.put("kind", e.kind());
                    item.set("payload", MAPPER.readTree(e.payload()));
                }
                response.set("memory", session.memorySnapshot());
            } else {
                response.putNull("sessionId");
                response.set("memory", emptyMemory());
            }
            sendJson(exchange, 200, response.toString());
        } catch (Exception e) {
            LOG.log(Level.WARNING, "Agent v2 history failed", e);
            sendQuietly(exchange, 500, errorJson(e.getMessage()));
        }
    }

    private void handleNewDialog(HttpExchange exchange) throws IOException {
        try {
            if (!requireMethod(exchange, "POST")) return;
            JsonNode body = readJson(exchange);
            String sid = UUID.randomUUID().toString();
            store.createDialog(sid, body.path("title").asText("Диалог"), body.path("profileId").asText(""));
            store.logAction(sid, "new_task", MAPPER.createObjectNode()
                    .put("profileId", body.path("profileId").asText("")).toString());
            ObjectNode response = MAPPER.createObjectNode();
            response.put("sessionId", sid);
            response.put("title", store.dialogTitle(sid));
            response.set("memory", sessionFor(sid, defaultClient).memorySnapshot());
            sendJson(exchange, 200, response.toString());
        } catch (IllegalArgumentException e) {
            sendQuietly(exchange, 400, errorJson(e.getMessage()));
        } catch (Exception e) {
            sendQuietly(exchange, 500, errorJson(e.getMessage()));
        }
    }

    private void handleDialogProfile(HttpExchange exchange) throws IOException {
        try {
            if (!requireMethod(exchange, "POST")) return;
            JsonNode body = readJson(exchange);
            String sid = body.path("dialogId").asText("").trim();
            String profileId = body.path("profileId").asText("").trim();
            if (!store.dialogExists(sid)) throw new IllegalArgumentException("Dialog not found.");
            if (!store.loadEntries(sid, "history").isEmpty()) {
                throw new IllegalArgumentException("Профиль можно выбрать только до первого сообщения.");
            }
            store.applyProfile(sid, profileId);
            store.logAction(sid, "profile_applied", MAPPER.createObjectNode().put("profileId", profileId).toString());
            sendJson(exchange, 200, memoryResponse(sid));
        } catch (IllegalArgumentException e) {
            sendQuietly(exchange, 400, errorJson(e.getMessage()));
        } catch (Exception e) {
            sendQuietly(exchange, 500, errorJson(e.getMessage()));
        }
    }

    private void handleProfiles(HttpExchange exchange) throws IOException {
        try {
            switch (exchange.getRequestMethod().toUpperCase()) {
                case "GET" -> {
                    ObjectNode response = MAPPER.createObjectNode();
                    response.set("profiles", store.profiles());
                    sendJson(exchange, 200, response.toString());
                }
                case "POST", "PUT" -> {
                    JsonNode body = readJson(exchange);
                    String id = body.path("id").asText("").trim();
                    String name = body.path("name").asText("").trim();
                    if (name.isEmpty()) throw new IllegalArgumentException("Название профиля не должно быть пустым.");
                    if (name.length() > 80) throw new IllegalArgumentException("Название профиля слишком длинное.");
                    ObjectNode existing = id.isEmpty() ? null : store.profile(id);
                    if (existing != null && existing.path("builtin").asBoolean(false)) {
                        throw new IllegalArgumentException("Встроенный профиль нельзя изменить; сохраните его копию.");
                    }
                    ObjectNode config = validateConfig(body);
                    if (id.isEmpty()) id = "profile-" + UUID.randomUUID();
                    store.saveProfile(id, name, config.path("rules").toString(), false);
                    ObjectNode response = MAPPER.createObjectNode();
                    response.put("profileId", id);
                    response.set("profiles", store.profiles());
                    sendJson(exchange, 200, response.toString());
                }
                case "DELETE" -> {
                    String id = queryParam(exchange, "id");
                    ObjectNode existing = store.profile(id);
                    if (existing == null) throw new IllegalArgumentException("Профиль не найден.");
                    if (existing.path("builtin").asBoolean(false)) {
                        throw new IllegalArgumentException("Встроенный профиль нельзя удалить.");
                    }
                    store.deleteProfile(id);
                    ObjectNode response = MAPPER.createObjectNode();
                    response.set("profiles", store.profiles());
                    sendJson(exchange, 200, response.toString());
                }
                default -> sendJson(exchange, 405, errorJson("Use GET, POST, PUT or DELETE."));
            }
        } catch (IllegalArgumentException e) {
            sendQuietly(exchange, 400, errorJson(e.getMessage()));
        } catch (Exception e) {
            LOG.log(Level.WARNING, "Agent v2 profile operation failed", e);
            sendQuietly(exchange, 500, errorJson(e.getMessage()));
        }
    }

    private void handleInvariants(HttpExchange exchange) throws IOException {
        try {
            switch (exchange.getRequestMethod().toUpperCase()) {
                case "GET" -> sendJson(exchange, 200, invariantsResponse().toString());
                case "POST", "PUT" -> {
                    JsonNode body = readJson(exchange);
                    String id = body.path("id").asText("").trim();
                    String text = body.path("text").asText("").trim();
                    if (text.isEmpty()) throw new IllegalArgumentException("Текст инварианта не должен быть пустым.");
                    if (text.length() > 4000) throw new IllegalArgumentException("Текст инварианта слишком длинный.");
                    String checkType = body.path("checkType").asText("semantic");
                    if (!Set.of("semantic", "deterministic").contains(checkType)) {
                        throw new IllegalArgumentException("Тип проверки: semantic или deterministic.");
                    }
                    JsonNode config = body.path("config");
                    if (!config.isObject()) throw new IllegalArgumentException("config должен быть JSON-объектом.");
                    ObjectNode value = MAPPER.createObjectNode();
                    value.put("id", id.isEmpty() ? "inv-" + UUID.randomUUID() : id);
                    value.put("category", body.path("category").asText("constraint").trim());
                    value.put("text", text);
                    value.put("checkType", checkType);
                    value.set("config", config);
                    value.put("enabled", body.path("enabled").asBoolean(true));
                    store.saveInvariant(value, "explicit_editor", body.path("dialogId").asText(""));
                    sendJson(exchange, 200, invariantsResponse().toString());
                }
                case "DELETE" -> {
                    String id = queryParam(exchange, "id");
                    String sid = queryParam(exchange, "dialog");
                    store.deleteInvariant(id, "explicit_editor", sid);
                    sendJson(exchange, 200, invariantsResponse().toString());
                }
                default -> sendJson(exchange, 405, errorJson("Use GET, POST, PUT or DELETE."));
            }
        } catch (IllegalArgumentException e) {
            sendQuietly(exchange, 400, errorJson(e.getMessage()));
        } catch (Exception e) {
            LOG.log(Level.WARNING, "Agent v2 invariant operation failed", e);
            sendQuietly(exchange, 500, errorJson(e.getMessage()));
        }
    }

    private ObjectNode invariantsResponse() {
        ObjectNode response = MAPPER.createObjectNode();
        response.set("invariants", store.invariants(false));
        response.set("invariantHistory", store.invariantHistory(50));
        response.put("invariantRevision", store.invariantRevision());
        return response;
    }

    private void handleMemory(HttpExchange exchange) throws IOException {
        try {
            String sid = dialogOrCurrent(exchange);
            String layer = queryParam(exchange, "layer");
            if (sid == null && !"lt".equals(layer)) {
                sendJson(exchange, 400, errorJson("No dialog selected."));
                return;
            }
            switch (exchange.getRequestMethod().toUpperCase()) {
                case "GET" -> {
                    ObjectNode response = MAPPER.createObjectNode();
                    if ("lt".equals(layer)) {
                        response.set("longTerm", sid == null ? store.agentConfig() : store.dialogAgentConfig(sid));
                    } else if ("wt".equals(layer)) {
                        response.set("working", store.working(sid));
                    } else if ("st".equals(layer)) {
                        response.set("facts", factsJson(sid));
                    } else {
                        response.set("memory", sessionFor(sid, defaultClient).memorySnapshot());
                    }
                    sendJson(exchange, 200, response.toString());
                }
                case "PUT" -> {
                    JsonNode body = readJson(exchange);
                    if ("lt".equals(layer)) {
                        ObjectNode config = validateConfig(body.has("longTerm") ? body.get("longTerm") : body);
                        if (sid == null) store.saveAgentConfig(config.toString());
                        else store.saveDialogAgentConfig(sid, config.toString());
                    } else if ("wt".equals(layer)) {
                        JsonNode working = body.has("working") ? body.get("working") : body;
                        store.saveWorking(sid, working);
                    } else if ("st".equals(layer)) {
                        List<AgentProtocol.Fact> facts = AgentProtocol.parseFacts(body.toString());
                        store.replaceFacts(sid, facts);
                        AgentV2Session active = sessions.get(sid);
                        if (active != null) active.restoreFacts(store.listFacts(sid));
                    } else {
                        sendJson(exchange, 400, errorJson("Unknown memory layer."));
                        return;
                    }
                    store.logAction(sid, "memory_edit", MAPPER.createObjectNode().put("layer", layer).toString());
                    sendJson(exchange, 200, memoryResponse(sid));
                }
                case "DELETE" -> {
                    if ("lt".equals(layer)) {
                        if (sid == null) store.saveAgentConfig("{\"rules\":[]}");
                        else store.saveDialogAgentConfig(sid, "{\"rules\":[]}");
                        store.setState("lt_initialized", "true");
                        if (sid != null) store.setState("lt_bootstrap_completed:" + sid, "true");
                    } else if ("wt".equals(layer)) {
                        store.clearWorking(sid);
                    } else if ("st".equals(layer)) {
                        store.clearShortTerm(sid);
                        AgentV2Session removed = sessions.remove(sid);
                        if (removed != null) removed.abort();
                    } else {
                        sendJson(exchange, 400, errorJson("Unknown memory layer."));
                        return;
                    }
                    store.logAction(sid, "memory_clear", MAPPER.createObjectNode().put("layer", layer).toString());
                    sendJson(exchange, 200, memoryResponse(sid));
                }
                default -> sendJson(exchange, 405, errorJson("Use GET, PUT or DELETE."));
            }
        } catch (IllegalArgumentException e) {
            sendQuietly(exchange, 400, errorJson(e.getMessage()));
        } catch (Exception e) {
            LOG.log(Level.WARNING, "Agent v2 memory operation failed", e);
            sendQuietly(exchange, 500, errorJson(e.getMessage()));
        }
    }

    private void handleRuleCandidate(HttpExchange exchange) throws IOException {
        try {
            if (!requireMethod(exchange, "POST")) return;
            JsonNode body = readJson(exchange);
            long id = body.path("id").asLong(0);
            String action = body.path("action").asText("");
            ObjectNode candidate = store.ruleCandidate(id);
            if (candidate == null || !"pending".equals(candidate.path("status").asText())) {
                sendJson(exchange, 404, errorJson("Rule candidate not found or already resolved."));
                return;
            }
            String sid = candidate.path("sourceDialog").asText("");
            if ("approve".equals(action)) {
                String text = body.path("text").asText(candidate.path("text").asText()).trim();
                if (text.isEmpty()) throw new IllegalArgumentException("Rule text must not be empty.");
                ObjectNode config = store.dialogAgentConfig(sid);
                ArrayNode rules = config.withArray("rules");
                for (JsonNode rule : rules) {
                    boolean updateTarget = "update".equals(candidate.path("operation").asText())
                            && candidate.path("targetRuleId").asText().equals(rule.path("id").asText());
                    if (!updateTarget && rule.path("text").asText("").trim().equalsIgnoreCase(text)) {
                        throw new IllegalArgumentException("Такое правило уже есть в Long-term.");
                    }
                }
                boolean updated = false;
                if ("update".equals(candidate.path("operation").asText())) {
                    String target = candidate.path("targetRuleId").asText();
                    for (JsonNode item : rules) {
                        if (target.equals(item.path("id").asText())
                                && candidate.path("baseText").asText().equals(item.path("text").asText())) {
                            ((ObjectNode) item).put("text", text);
                            updated = true;
                            break;
                        }
                    }
                }
                if (!updated) {
                    // Missing or manually edited update target: add instead of overwriting it.
                    String ruleId = "preference-" + candidate.path("key").asText();
                    int suffix = 2;
                    Set<String> ids = new HashSet<>();
                    rules.forEach(item -> ids.add(item.path("id").asText()));
                    while (ids.contains(ruleId)) ruleId = "preference-" + candidate.path("key").asText() + "-" + suffix++;
                    ObjectNode rule = rules.addObject();
                    rule.put("id", ruleId);
                    rule.put("type", "preference");
                    rule.put("text", text);
                    rule.put("enabled", true);
                    rule.put("sourceFact", candidate.path("sourceFact").asText());
                }
                validateConfig(config);
                store.saveDialogAgentConfig(sid, config.toString());
                store.resolveRuleCandidate(id, "approved");
            } else if ("reject".equals(action)) {
                store.resolveRuleCandidate(id, "rejected");
            } else {
                sendJson(exchange, 400, errorJson("Use action approve or reject."));
                return;
            }
            store.logAction(sid, "rule_candidate_" + action,
                    MAPPER.createObjectNode().put("candidateId", id).toString());
            sendJson(exchange, 200, memoryResponse(sid));
        } catch (IllegalArgumentException e) {
            sendQuietly(exchange, 400, errorJson(e.getMessage()));
        } catch (Exception e) {
            LOG.log(Level.WARNING, "Failed to resolve LT rule candidate", e);
            sendQuietly(exchange, 500, errorJson(e.getMessage()));
        }
    }

    private ObjectNode validateConfig(JsonNode config) {
        if (config == null || !config.isObject() || !config.path("rules").isArray()) {
            throw new IllegalArgumentException("LT config must be an object with a rules array.");
        }
        Set<String> ids = new HashSet<>();
        Set<String> types = Set.of("role", "behavior", "output_format", "tool_policy", "constraint", "preference", "custom");
        for (JsonNode rule : config.path("rules")) {
            String id = rule.path("id").asText("").trim();
            String type = rule.path("type").asText("").trim();
            String text = rule.path("text").asText("").trim();
            if (id.isEmpty() || !ids.add(id)) throw new IllegalArgumentException("Rule ids must be non-empty and unique.");
            if (!types.contains(type)) throw new IllegalArgumentException("Unknown rule type: " + type);
            if (text.isEmpty()) throw new IllegalArgumentException("Rule text must not be empty.");
        }
        return (ObjectNode) config;
    }

    private String memoryResponse(String sid) {
        ObjectNode response = MAPPER.createObjectNode();
        response.set("memory", sid == null ? emptyMemory() : sessionFor(sid, defaultClient).memorySnapshot());
        return response.toString();
    }

    private void handleReset(HttpExchange exchange) throws IOException {
        try {
            if (!requireMethod(exchange, "POST")) return;
            JsonNode body = readJson(exchange);
            if (!body.path("confirm").asBoolean(false)) {
                sendJson(exchange, 400, errorJson("Pass confirm=true to reset all memory."));
                return;
            }
            sessions.values().forEach(AgentV2Session::abort);
            sessions.clear();
            store.resetAll();
            sendJson(exchange, 200, "{\"ok\":true}");
        } catch (Exception e) {
            sendQuietly(exchange, 500, errorJson(e.getMessage()));
        }
    }

    private void handleLlmLog(HttpExchange exchange) throws IOException {
        try {
            if (!requireMethod(exchange, "GET")) return;
            ObjectNode response = MAPPER.createObjectNode();
            response.set("items", store.llmLogs(dialogOrCurrent(exchange)));
            sendJson(exchange, 200, response.toString());
        } catch (Exception e) {
            sendQuietly(exchange, 500, errorJson(e.getMessage()));
        }
    }

    private void handleActionLog(HttpExchange exchange) throws IOException {
        try {
            if (!requireMethod(exchange, "GET")) return;
            ObjectNode response = MAPPER.createObjectNode();
            response.set("items", store.actionLogs(dialogOrCurrent(exchange)));
            sendJson(exchange, 200, response.toString());
        } catch (Exception e) {
            sendQuietly(exchange, 500, errorJson(e.getMessage()));
        }
    }

    private void handleAction(HttpExchange exchange) throws IOException {
        try {
            if (!requireMethod(exchange, "POST")) return;
            JsonNode body = readJson(exchange);
            String action = body.path("action").asText("").trim();
            if (action.isEmpty()) {
                sendJson(exchange, 400, errorJson("Action is required."));
                return;
            }
            String sid = body.path("sessionId").asText(store.currentDialogId());
            JsonNode details = body.path("details");
            store.logAction(sid, action, details.isObject() ? details.toString() : "{}");
            sendJson(exchange, 200, "{\"ok\":true}");
        } catch (Exception e) {
            sendQuietly(exchange, 500, errorJson(e.getMessage()));
        }
    }

    private void handleAnalyze(HttpExchange exchange) throws IOException {
        String sid = null;
        try {
            if (!requireMethod(exchange, "POST")) return;
            JsonNode body = readJson(exchange);
            String scope = body.path("scope").asText("dialog");
            List<String> dialogIds = new ArrayList<>();
            if ("script".equals(scope)) {
                for (JsonNode id : body.path("dialogIds")) {
                    String value = id.asText("").trim();
                    if (!value.isEmpty() && !dialogIds.contains(value)) dialogIds.add(value);
                }
            } else {
                String value = body.path("dialogId").asText(store.currentDialogId());
                if (value != null && !value.isBlank()) dialogIds.add(value);
            }
            if (dialogIds.isEmpty()) {
                sendJson(exchange, 400, errorJson("No dialogs selected for analysis."));
                return;
            }
            for (String dialogId : dialogIds) {
                if (!store.dialogExists(dialogId)) {
                    sendJson(exchange, 400, errorJson("Dialog not found: " + dialogId));
                    return;
                }
            }
            sid = dialogIds.get(0);
            LlmClient client = resolveClient(body.path("model").asText(""));
            if (client == null) {
                sendJson(exchange, 400, errorJson("Unknown analysis model."));
                return;
            }
            String prompt = analysisPrompt(dialogIds, scope);
            ObjectNode request = MAPPER.createObjectNode();
            request.put("model", client.model());
            request.put("prompt", prompt);
            long started = System.nanoTime();
            LlmClient.ChatResult result = client.chatOnce(List.of(new LlmClient.Message("user", prompt)),
                    new LlmClient.ChatOptions("none", "", null, List.of(), "none", null));
            long duration = (System.nanoTime() - started) / 1_000_000;
            ObjectNode usage = MAPPER.createObjectNode();
            if (result.promptTokens() != null) usage.put("promptTokens", result.promptTokens());
            if (result.completionTokens() != null) usage.put("completionTokens", result.completionTokens());
            if (result.totalTokens() != null) usage.put("totalTokens", result.totalTokens());
            if (result.reasoningTokens() != null) usage.put("reasoningTokens", result.reasoningTokens());
            store.logLlm(sid, "analysis", request.toString(), result.reply(), result.reasoning(),
                    "{}", usage.toString(), duration);
            ObjectNode analysisAction = MAPPER.createObjectNode();
            analysisAction.put("model", client.model());
            analysisAction.put("scope", scope);
            analysisAction.set("dialogIds", MAPPER.valueToTree(dialogIds));
            store.logAction(sid, "analysis", analysisAction.toString());
            ObjectNode response = MAPPER.createObjectNode();
            response.put("reply", result.reply());
            response.put("model", client.model());
            response.set("usage", usage);
            sendJson(exchange, 200, response.toString());
        } catch (Exception e) {
            LOG.log(Level.WARNING, "Agent v2 analysis failed", e);
            store.logAction(sid, "analysis_error", MAPPER.createObjectNode()
                    .put("error", e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage()).toString());
            sendQuietly(exchange, 500, errorJson("Analysis failed: " + e.getMessage()));
        }
    }

    private String analysisPrompt(List<String> dialogIds, String scope) {
        ObjectNode data = MAPPER.createObjectNode();
        data.put("scope", scope);
        ArrayNode dialogs = data.putArray("dialogs");
        for (String sid : dialogIds) {
            ObjectNode dialog = dialogs.addObject();
            dialog.put("dialogId", sid);
            dialog.put("title", store.dialogTitle(sid));
            dialog.put("profileId", store.dialogProfileId(sid));
            ObjectNode profile = store.profile(store.dialogProfileId(sid));
            dialog.put("profileName", profile == null ? "" : profile.path("name").asText());
            dialog.set("longTerm", store.dialogAgentConfig(sid));
            dialog.set("turns", analysisTurns(sid));
            dialog.set("tokenSummary", analysisTokenSummary(sid));
            dialog.set("technicalCalls", analysisTechnicalCalls(sid));
            dialog.set("working", store.working(sid));
            dialog.set("facts", factsJson(sid));
            dialog.set("userActions", analysisUserActions(sid));
        }
        if ("script".equals(scope)) {
            return """
                    Ты проверяешь результаты УЖЕ ВЫПОЛНЕННОГО скрипта: насколько точно агент
                    соблюдает профиль каждого диалога. Не анализируй интерфейс или код приложения.
                    Отвечай по-русски в корректном Markdown.

                    Для каждого диалога используй profileName и включённые правила longTerm как эталон.
                    Проверь ВСЕ основные ответы assistant из turns, а не только первый или последний.
                    Отдельно оцени:
                    - соблюдение требований к подробности, структуре и тону;
                    - наличие или отсутствие кода в соответствии с профилем;
                    - стабильность соблюдения профиля от ответа к ответу;
                    - техническую корректность ответа отдельно от соответствия профилю.

                    Структура результата:
                    1. «Итог по профилям» — таблица: профиль, вердикт (соблюдён / частично /
                       нарушён), какие ответы содержат отклонения, краткое обоснование.
                    2. «Проверка правил» — для каждого профиля отдельно проверь каждое включённое
                       правило и приведи короткие примеры из конкретных номеров ответов.
                    3. «Сравнение одинаковых вопросов» — выясни, действительно ли ответы на одинаковые
                       вопросы различаются согласно профилям. Не считай нарушением различия, которые
                       не связаны с правилами профиля.
                    4. «Технические ошибки» — отдельно перечисли ошибки в содержании, если они есть.
                    5. «Вывод» — кратко укажи, какой профиль соблюдается лучше и что следует изменить
                       в правилах или системном промпте.

                    Не повторяй ответы целиком и не пересказывай вопросы. Ссылайся на номера turns.
                    Не выдумывай отклонения: если профиль соблюдён, так и напиши. Если ответов нет
                    или скрипт не выполнен полностью, явно отметь недостаток данных.

                    Результаты выполненного скрипта:
                    %s
                    """.formatted(data.toString());
        }
        return """
                Ты анализируешь ПОВЕДЕНИЕ ПОЛЬЗОВАТЕЛЯ И ПАМЯТИ АГЕНТА, а не функции
                приложения и не качество его интерфейса. Используй транскрипт, журнал действий,
                текущие слои и LLM-журнал. Отвечай по-русски в корректном Markdown.

                Обязательная структура ответа:
                1. «Действия пользователя» — хронология сообщений, смены настроек, ручных правок,
                   очисток и загрузок; выдели намерения пользователя и изменения требований.
                2. «Что попадает в слои» — конкретные данные из этого диалога в ST, sticky-фактах,
                   WT и LT; почему они там оказались, есть ли ошибочные, лишние или потерянные данные.
                3. «Вклад слоёв в токены» — таблица System / LT / WT / ST / User по конкретным
                   запросам: используй фактические usage и layer_stats из журнала. Отдельно отмечай
                   оценённые значения; не выдумывай отсутствующие числа.
                4. «Последовательная очистка слоёв» — сценарии очистки ST, затем sticky-фактов,
                   затем WT, затем LT. Для каждого шага укажи, какие именно данные исчезнут из
                   контекста и на какие РЕАЛЬНЫЕ вопросы из этого диалога агент, вероятно,
                   перестанет отвечать корректно. Приводи цитаты вопросов. Это прогноз по данным,
                   а не результат повторного запуска — явно обозначь это.
                5. «Проблемы и рекомендации» — потерянные ответы, противоречия, лишний расход
                   токенов и конкретные настройки памяти/сжатия.

                Не делай обзор возможностей приложения, не оценивай его UI и не описывай код.
                Требуются только краткие выводы по имеющимся данным: не повторяй исходные JSON,
                не пересказывай диалог и не приводи длинные цитаты. Метрики пиши компактно в одну
                строку или в таблицу, никогда не выноси знаки препинания на отдельные строки.
                Для scope=script дополнительно сравни поведение и память диалогов между собой.
                Если данных для вывода недостаточно, так и напиши.

                Данные для анализа:
                %s
                """.formatted(data.toString());
    }

    /** Compact turn view: no repeated request messages, reasoning or raw technical payloads. */
    private ArrayNode analysisTurns(String sid) {
        ArrayNode turns = MAPPER.createArrayNode();
        ArrayNode logs = store.llmLogs(sid);
        int number = 0;
        for (int i = logs.size() - 1; i >= 0; i--) {
            ObjectNode log = (ObjectNode) logs.get(i);
            if (!"chat".equals(log.path("kind").asText())) continue;
            ObjectNode turn = turns.addObject();
            turn.put("number", ++number);
            String user = "";
            for (JsonNode message : log.path("request").path("messages")) {
                if ("user".equals(message.path("role").asText())) user = message.path("content").asText("");
            }
            turn.put("user", abbreviateMiddle(user, 2000));
            turn.put("assistant", abbreviateMiddle(log.path("response").asText(""), 2000));
            turn.set("layerTokens", log.path("layers"));
            turn.set("usage", log.path("usage"));
            turn.put("durationMs", log.path("durationMs").asLong());
        }
        return turns;
    }

    private ObjectNode analysisTokenSummary(String sid) {
        ObjectNode summary = MAPPER.createObjectNode();
        ObjectNode layers = summary.putObject("estimatedLayerTokens");
        int prompt = 0;
        int completion = 0;
        int total = 0;
        int calls = 0;
        ArrayNode logs = store.llmLogs(sid);
        for (JsonNode log : logs) {
            if (!"chat".equals(log.path("kind").asText())) continue;
            calls++;
            for (String layer : List.of("system", "longTerm", "working", "shortTerm")) {
                String key = switch (layer) {
                    case "longTerm" -> "longTerm"; case "working" -> "working";
                    case "shortTerm" -> "shortTerm"; default -> "system";
                };
                layers.put(layer, layers.path(layer).asInt(0)
                        + log.path("layers").path(key).path("estTokens").asInt(0));
            }
            prompt += log.path("usage").path("promptTokens").asInt(0);
            completion += log.path("usage").path("completionTokens").asInt(0);
            total += log.path("usage").path("totalTokens").asInt(0);
        }
        summary.put("mainCalls", calls);
        summary.put("promptTokens", prompt);
        summary.put("completionTokens", completion);
        summary.put("totalTokens", total);
        return summary;
    }

    private ArrayNode analysisTechnicalCalls(String sid) {
        ArrayNode result = MAPPER.createArrayNode();
        ArrayNode logs = store.llmLogs(sid);
        for (int i = logs.size() - 1; i >= 0; i--) {
            ObjectNode log = (ObjectNode) logs.get(i);
            String kind = log.path("kind").asText();
            if ("chat".equals(kind) || "analysis".equals(kind)) continue;
            ObjectNode item = result.addObject();
            item.put("kind", kind);
            item.put("durationMs", log.path("durationMs").asLong());
            item.set("usage", log.path("usage"));
            item.put("hasResult", !log.path("response").asText("").isBlank());
        }
        return result;
    }

    private ArrayNode analysisUserActions(String sid) {
        ArrayNode result = MAPPER.createArrayNode();
        ArrayNode logs = store.actionLogs(sid);
        int added = 0;
        for (int i = logs.size() - 1; i >= 0 && added < 100; i--) {
            ObjectNode log = (ObjectNode) logs.get(i);
            String action = log.path("action").asText();
            if ("message_sent".equals(action) || "analysis".equals(action)) continue;
            ObjectNode item = result.addObject();
            item.put("action", action);
            item.put("createdAt", log.path("createdAt").asText());
            item.set("details", log.path("details"));
            added++;
        }
        return result;
    }

    private static String abbreviateMiddle(String text, int max) {
        if (text == null || text.length() <= max) return text == null ? "" : text;
        int side = max / 2;
        return text.substring(0, side) + "\n…[сокращено " + (text.length() - max) + " символов]…\n"
                + text.substring(text.length() - side);
    }

    private ObjectNode emptyMemory() {
        ObjectNode memory = MAPPER.createObjectNode();
        String invariantPrompt = store.invariantPrompt();
        memory.put("systemPrompt", AgentProtocol.systemPrompt(false)
                + (invariantPrompt.isEmpty() ? "" : "\n" + invariantPrompt));
        memory.set("invariants", store.invariants(false));
        memory.set("invariantHistory", store.invariantHistory(50));
        memory.put("invariantRevision", store.invariantRevision());
        memory.set("longTerm", store.agentConfig());
        memory.set("ruleCandidates", MAPPER.createArrayNode());
        memory.set("working", MAPPER.createObjectNode());
        memory.set("facts", MAPPER.createArrayNode());
        memory.set("layers", MAPPER.createObjectNode());
        return memory;
    }

    private ArrayNode factsJson(String sid) {
        ArrayNode facts = MAPPER.createArrayNode();
        for (AgentProtocol.Fact f : store.listFacts(sid)) {
            facts.addObject().put("category", f.category()).put("key", f.key()).put("value", f.value());
        }
        return facts;
    }

    private String dialogOrCurrent(HttpExchange exchange) {
        String sid = queryParam(exchange, "dialog");
        return sid == null || sid.isBlank() ? store.currentDialogId() : sid;
    }

    private LlmClient resolveClient(String ref) {
        if (ref == null || ref.isBlank()) return defaultClient;
        for (AppConfig.ModelConfig c : modelConfigs) {
            if (c.id().equalsIgnoreCase(ref) || c.model().equalsIgnoreCase(ref)) {
                return clientsById.get(c.id());
            }
        }
        return null;
    }

    private Integer contextWindow(LlmClient client) {
        for (AppConfig.ModelConfig c : modelConfigs) {
            if (clientsById.get(c.id()) == client) return c.contextWindow();
        }
        return null;
    }

    private static JsonNode readJson(HttpExchange exchange) throws IOException {
        return MAPPER.readTree(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
    }

    private static JsonNode readTree(String raw) {
        try {
            return MAPPER.readTree(raw);
        } catch (Exception e) {
            return MAPPER.createObjectNode().put("raw", raw);
        }
    }

    private static boolean requireMethod(HttpExchange exchange, String method) throws IOException {
        if (method.equalsIgnoreCase(exchange.getRequestMethod())) return true;
        sendJson(exchange, 405, errorJson("Method not allowed. Use " + method + "."));
        return false;
    }

    private static String queryParam(HttpExchange exchange, String name) {
        String query = exchange.getRequestURI().getRawQuery();
        if (query == null) return null;
        for (String pair : query.split("&")) {
            int eq = pair.indexOf('=');
            if (eq > 0 && pair.substring(0, eq).equals(name)) {
                return java.net.URLDecoder.decode(pair.substring(eq + 1), StandardCharsets.UTF_8);
            }
        }
        return null;
    }

    private static void writeSse(OutputStream out, String event, String data) throws IOException {
        out.write(("event: " + event + "\n").getBytes(StandardCharsets.UTF_8));
        out.write(("data: " + data + "\n\n").getBytes(StandardCharsets.UTF_8));
        out.flush();
    }

    private static ObjectNode errorJson(String message) {
        ObjectNode error = MAPPER.createObjectNode();
        error.put("error", message == null ? "Unknown error" : message);
        return error;
    }

    private static void sendJson(HttpExchange exchange, int status, ObjectNode json) throws IOException {
        sendJson(exchange, status, json.toString());
    }

    private static void sendJson(HttpExchange exchange, int status, String json) throws IOException {
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        exchange.sendResponseHeaders(status, bytes.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }

    private static void sendQuietly(HttpExchange exchange, int status, ObjectNode json) {
        try {
            sendJson(exchange, status, json);
        } catch (Exception ignored) {
            exchange.close();
        }
    }

    private static String abbreviate(String text, int max) {
        String oneLine = text == null ? "" : text.replaceAll("\\s+", " ").trim();
        return oneLine.length() <= max ? oneLine : oneLine.substring(0, max) + "…";
    }
}
