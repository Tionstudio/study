package ru.tionstudio.ai.advent.challenge.agent;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import ru.tionstudio.ai.advent.challenge.llm.LlmClient;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * One agent chat session: keeps the dialog history, runs the
 * "plan → LLM → tool call(s) → tool result(s) → LLM …" loop and enforces
 * the safety rules:
 * <ul>
 *   <li>the system prompt (tool contract) is pinned at the head of the context
 *       and is never compressed;</li>
 *   <li>context compression: "none" or "lastN" — the window counts only
 *       user/assistant messages, tool records are kept with their turn;</li>
 *   <li>tool calls are validated before execution;</li>
 *   <li>write/append/delete require explicit per-operation user permission;</li>
 *   <li>the number of consecutive tool calls is limited (default 5);</li>
 *   <li>independent calls from one reply run in parallel (virtual threads);</li>
 *   <li>the user may pause processing between agent steps.</li>
 * </ul>
 */
public class AgentSession {

    /** Receives SSE-ready events produced while a turn is running. */
    public interface Listener {
        void on(String event, ObjectNode data) throws IOException;
    }

    /**
     * Persistence hook: everything needed to re-render the chat (transcript)
     * and to rebuild the LLM context (history) after an application restart.
     * Implemented by the server wiring; failures must not break the turn.
     */
    public interface TranscriptListener {
        void onEntry(String kind, ObjectNode payload);

        void onHistory(String role, String content);
    }

    /** One LLM-context entry for session restore (role: user|assistant|tool). */
    public record HistoryEntry(String role, String content) {
    }

    /** Session settings sent by the page with every message. */
    public record Settings(String compression, int contextMessages,
                           int maxToolCalls, boolean planning) {
    }

    private record Entry(String role, String content) {
    }

    /** One plan step as shown in the UI. */
    private record PlanStep(String title, String status) {
    }

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final Logger LOG = Logger.getLogger(AgentSession.class.getName());

    /** Permission requests time out after this long and count as denied. */
    private static final long PERMISSION_TIMEOUT_MINUTES = 5;

    private final String id;
    private final LlmClient client;
    private final AgentTools tools;
    /** Shared registry: permission request id → decision future. */
    private final Map<String, CompletableFuture<Boolean>> permissionRegistry;
    /** Parallel tool execution (virtual threads are cheap to block). */
    private final ExecutorService toolExecutor = Executors.newVirtualThreadPerTaskExecutor();

    private final List<Entry> history = new ArrayList<>();
    private String compression = "none";
    private int contextMessages = 20;
    private int maxToolCalls = 5;
    private boolean planning = true;

    /** Persistence hook (set by the server when the session is created). */
    private volatile TranscriptListener transcript;

    /**
     * Cumulative token usage across ALL turns of this dialog
     * (-1 = no usage reported yet). Restored from the store after a restart.
     */
    private int sessionPromptTokens = -1;
    private int sessionCompletionTokens = 0;

    /** Start of the current turn (for the duration shown in the status). */
    private long turnStartNanos;

    /** Model context window in tokens (0 = unknown, no fill warnings). */
    private int contextWindow;
    /** Highest context-fill warning already shown (0, 1 = 85%, 2 = 95%). */
    private int contextWarnedLevel;

    /** Current turn's plan (reset on every user message). */
    private List<PlanStep> plan = new ArrayList<>();

    private volatile boolean pauseRequested;
    private volatile boolean aborted;
    /** Body stream of the in-flight LLM request — closed by abort(). */
    private volatile AutoCloseable activeLlmConnection;
    private final Object pauseLock = new Object();
    /** Decisions this session is currently waiting for (permission/limit). */
    private final List<CompletableFuture<Boolean>> pendingDecisions =
            java.util.Collections.synchronizedList(new ArrayList<>());

    public AgentSession(String id, LlmClient client, AgentTools tools,
                        Map<String, CompletableFuture<Boolean>> permissionRegistry) {
        this.id = id;
        this.client = client;
        this.tools = tools;
        this.permissionRegistry = permissionRegistry;
    }

    public String id() {
        return id;
    }

    public void setTranscriptListener(TranscriptListener listener) {
        this.transcript = listener;
    }

    /** Rebuilds the LLM context from entries stored before a restart. */
    public void restoreHistory(List<HistoryEntry> entries) {
        history.clear();
        for (HistoryEntry e : entries) {
            history.add(new Entry(e.role(), e.content()));
        }
        LOG.info("Agent session " + id + ": history restored, "
                + history.size() + " entries");
    }

    /** Restores the cumulative dialog token usage stored before a restart. */
    public void restoreUsage(int promptTokens, int completionTokens) {
        this.sessionPromptTokens = promptTokens;
        this.sessionCompletionTokens = completionTokens;
    }

    /** Sets the model context window (tokens) for fill-level warnings. */
    public void setContextWindow(Integer window) {
        this.contextWindow = window == null ? 0 : Math.max(0, window);
    }

    /**
     * Warns in the chat when the prompt approaches the model's context
     * window: at 85% and again at 95% (each once, re-armed when the fill
     * drops below 80% — e.g. after compression starts dropping messages).
     */
    private void checkContextFill(Integer promptTokens, Listener listener) throws IOException {
        if (contextWindow <= 0 || promptTokens == null) {
            return;
        }
        int pct = (int) (promptTokens * 100L / contextWindow);
        if (pct < 80) {
            contextWarnedLevel = 0;
            return;
        }
        int level = pct >= 95 ? 2 : (pct >= 85 ? 1 : 0);
        if (level <= contextWarnedLevel) {
            return;
        }
        contextWarnedLevel = level;
        String text = level >= 2
                ? "⚠️ Контекстное окно модели почти исчерпано: " + promptTokens + " из "
                        + contextWindow + " токенов (" + pct + "%). Ответы вот-вот начнут "
                        + "обрезаться — начните новый диалог или включите сжатие контекста."
                : "Контекстное окно модели заполнено на " + pct + "% (" + promptTokens + " из "
                        + contextWindow + " токенов). При дальнейшем росте ответы начнут "
                        + "обрезаться — включите сжатие контекста.";
        ObjectNode note = MAPPER.createObjectNode();
        note.put("text", text);
        if (level >= 2) {
            // The chat shows a "Сжать контекст" button next to this note
            note.put("offerCompression", true);
        }
        listener.on("note", note);
        emitTranscript("note", note);
        LOG.warning("Agent session " + id + ": context window " + pct + "% full ("
                + promptTokens + "/" + contextWindow + ")");
    }

    /** Persists a transcript entry; persistence errors are logged, not thrown. */
    private void emitTranscript(String kind, ObjectNode payload) {
        TranscriptListener t = transcript;
        if (t == null) {
            return;
        }
        try {
            t.onEntry(kind, payload);
        } catch (Exception e) {
            LOG.log(Level.WARNING, "Agent session " + id
                    + ": failed to persist transcript entry " + kind, e);
        }
    }

    /** Persists one LLM-context entry; errors are logged, not thrown. */
    private void emitHistory(String role, String content) {
        TranscriptListener t = transcript;
        if (t == null) {
            return;
        }
        try {
            t.onHistory(role, content);
        } catch (Exception e) {
            LOG.log(Level.WARNING, "Agent session " + id
                    + ": failed to persist history entry", e);
        }
    }

    private void emitTranscriptText(String kind, String text) {
        ObjectNode json = MAPPER.createObjectNode();
        json.put("text", text);
        emitTranscript(kind, json);
    }

    public void updateSettings(Settings s) {
        if (s == null) {
            return;
        }
        if ("lastN".equals(s.compression()) || "none".equals(s.compression())) {
            this.compression = s.compression();
        }
        if (s.contextMessages() > 0) {
            this.contextMessages = Math.min(s.contextMessages(), 500);
        }
        if (s.maxToolCalls() > 0) {
            this.maxToolCalls = Math.min(s.maxToolCalls(), 50);
        }
        this.planning = s.planning();
    }

    /**
     * Switches this session to "last N messages" compression — invoked by the
     * "Сжать контекст" chat button when the model hits the token limit.
     * Applies from the next LLM call; the history itself is kept intact.
     */
    public void enableCompression() {
        this.compression = "lastN";
        LOG.info("Agent session " + id + ": compression enabled (last " + contextMessages
                + " messages) via the chat button");
    }

    public void pause() {
        pauseRequested = true;
    }

    public void resume() {
        pauseRequested = false;
        synchronized (pauseLock) {
            pauseLock.notifyAll();
        }
    }

    public void abort() {
        aborted = true;
        resume();
        // Cancel the in-flight LLM request (if any): closing the body stream
        // makes the blocked read fail immediately instead of hanging forever
        AutoCloseable conn = activeLlmConnection;
        if (conn != null) {
            try {
                conn.close();
            } catch (Exception ignored) {
                // best effort
            }
        }
        // Unblock permission/limit waits — otherwise abort() during a pending
        // decision would hang until the 5-minute timeout
        synchronized (pendingDecisions) {
            for (CompletableFuture<Boolean> f : pendingDecisions) {
                f.complete(false);
            }
        }
    }

    /**
     * Runs one user turn: appends the message, builds a plan (if enabled),
     * then loops LLM calls and tool executions until the model gives a final
     * text answer, the tool-call limit is hit, or the run is aborted.
     */
    public synchronized void runTurn(String userMessage, Listener listener) throws IOException {
        aborted = false;
        pauseRequested = false;
        activeLlmConnection = null;
        turnStartNanos = System.nanoTime();
        history.add(new Entry("user", userMessage));
        emitHistory("user", userMessage);
        emitTranscriptText("user", userMessage);
        plan = new ArrayList<>();

        int consecutiveToolCalls = 0;
        int totalToolCalls = 0;
        int totalPrompt = 0, totalCompletion = 0, llmCalls = 0;
        int lastCallCompletion = -1; // completion tokens of the latest LLM call
        boolean hasUsage = false;

        try {
            // ----- Planning phase -----
            if (planning && !aborted) {
                // Tell the page the planning phase started — its tokens are
                // NOT streamed, so without this event the chat shows nothing
                // until the finished plan arrives
                listener.on("planning", MAPPER.createObjectNode());
                // Planning is an internal phase: its tokens are NOT streamed
                // to the chat — only the parsed plan event reaches the page
                LlmClient.ChatResult planResult = callLlm(
                        List.of(new LlmClient.Message("user",
                                AgentProtocol.planningPrompt(userMessage))),
                        listener, false);
                if (planResult == null) {
                    if (aborted) {
                        emitDone(listener, totalPrompt, totalCompletion, hasUsage,
                                llmCalls, totalToolCalls, true, -1);
                    }
                    return; // error already reported / client gone / aborted
                }
                llmCalls++;
                if (planResult.promptTokens() != null) {
                    totalPrompt += planResult.promptTokens();
                    hasUsage = true;
                }
                if (planResult.completionTokens() != null) {
                    totalCompletion += planResult.completionTokens();
                    hasUsage = true;
                }
                List<String> steps = AgentProtocol.parsePlan(planResult.reply());
                plan = new ArrayList<>();
                for (String title : steps) {
                    plan.add(new PlanStep(title, "pending"));
                }
                emitPlan(listener, true);
                LOG.info("Agent session " + id + ": plan, " + plan.size() + " steps");
            }

            // ----- Execution loop -----
            while (true) {
                LlmClient.ChatResult result = callLlm(buildMessages(), listener, true);
                if (result == null) {
                    if (aborted) {
                        emitDone(listener, totalPrompt, totalCompletion, hasUsage,
                                llmCalls, totalToolCalls, true, -1);
                    }
                    return; // error already reported / client gone / aborted
                }
                llmCalls++;
                if (result.promptTokens() != null) {
                    totalPrompt += result.promptTokens();
                    hasUsage = true;
                }
                checkContextFill(result.promptTokens(), listener);
                if (result.completionTokens() != null) {
                    totalCompletion += result.completionTokens();
                    lastCallCompletion = result.completionTokens();
                    hasUsage = true;
                }

                String reply = result.reply() == null ? "" : result.reply().trim();
                AgentProtocol.ParseResult parsed = AgentProtocol.parseFull(reply);
                List<AgentProtocol.ToolCall> calls = parsed.calls();
                if (calls.isEmpty()) {
                    // Regular text — the final answer of this turn
                    String finalReply = reply;
                    if ("length".equals(result.finishReason())) {
                        // The model hit the token limit mid-generation (often
                        // mid JSON tool call). Cut the dangling tail and warn.
                        String cleaned = AgentProtocol.stripTruncatedTail(reply);
                        if (!cleaned.equals(reply)) {
                            LOG.info("Agent session " + id
                                    + ": reply truncated by token limit, dangling JSON tail removed ("
                                    + (reply.length() - cleaned.length()) + " chars)");
                            finalReply = cleaned;
                            ObjectNode replace = MAPPER.createObjectNode();
                            replace.put("text", finalReply);
                            listener.on("reply", replace);
                        }
                        String noteText = "Ответ обрезан: модель упёрлась в лимит токенов"
                                + " (контекстное окно заполнено). Включите сжатие контекста"
                                + " или начните новый диалог.";
                        ObjectNode note = MAPPER.createObjectNode();
                        note.put("text", noteText);
                        // The chat shows a "Сжать контекст" button next to this note
                        note.put("offerCompression", true);
                        listener.on("note", note);
                        emitTranscript("note", note);
                        LOG.warning("Agent session " + id
                                + ": reply truncated by token limit (finish_reason=length)");
                    }
                    history.add(new Entry("assistant", finalReply));
                    emitHistory("assistant", finalReply);
                    emitTranscriptText("assistant", finalReply);
                    markPlanDone();
                    emitPlan(listener);
                    emitDone(listener, totalPrompt, totalCompletion, hasUsage,
                            llmCalls, totalToolCalls, false, lastCallCompletion);
                    return;
                }

                // The model wants tools (one object or a parallel array).
                // Any prose it wrote around the call JSON is shown in the chat
                // instead of the raw payload bubble.
                String prose = parsed.prose();
                history.add(new Entry("assistant", reply));
                emitHistory("assistant", reply);
                if (prose != null && !prose.isBlank()) {
                    emitTranscriptText("prose", prose);
                }
                long realCalls = calls.stream()
                        .filter(c -> !"update_plan".equals(c.tool())).count();
                consecutiveToolCalls += (int) realCalls;
                totalToolCalls += (int) realCalls;
                if (consecutiveToolCalls > maxToolCalls) {
                    // Not a hard stop: ask the user whether to keep going —
                    // long honest tasks (e.g. "roll dice 10 times") legitimately
                    // need more than the default 5 calls
                    Boolean proceed = askContinue(listener);
                    if (proceed == null) {
                        return; // client went away
                    }
                    if (!proceed) {
                        ObjectNode json = MAPPER.createObjectNode();
                        json.put("max", maxToolCalls);
                        listener.on("limit", json);
                        emitTranscriptText("note", "Достигнут лимит вызовов инструментов подряд ("
                                + maxToolCalls + "). Агент остановлен — уточните запрос.");
                        LOG.warning("Agent session " + id + ": tool-call limit (" + maxToolCalls
                                + ") exceeded, user stopped the run");
                        emitDone(listener, totalPrompt, totalCompletion, hasUsage,
                                llmCalls, totalToolCalls, true, -1);
                        return;
                    }
                    LOG.info("Agent session " + id + ": tool-call limit (" + maxToolCalls
                            + ") exceeded, user allowed continuing — counter reset");
                    consecutiveToolCalls = 0;
                }

                for (int i = 0; i < calls.size(); i++) {
                    AgentProtocol.ToolCall call = calls.get(i);
                    ObjectNode callJson = MAPPER.createObjectNode();
                    callJson.put("index", i);
                    callJson.put("tool", call.tool());
                    callJson.set("args", call.args());
                    if (i == 0 && prose != null && !prose.isBlank()) {
                        callJson.put("prose", prose);
                    }
                    listener.on("tool_call", callJson);
                }
                LOG.info("Agent session " + id + ": " + calls.size() + " tool call(s): "
                        + calls.stream()
                                .map(c -> c.tool() + " " + abbreviate(c.args().toString(), 200))
                                .toList());

                // Independent calls run in parallel; permission decisions
                // simply block their own call until the user responds
                List<CompletableFuture<AgentTools.ToolResult>> futures = new ArrayList<>();
                for (int i = 0; i < calls.size(); i++) {
                    final int index = i;
                    final AgentProtocol.ToolCall call = calls.get(i);
                    futures.add(CompletableFuture.supplyAsync(() -> {
                        try {
                            return executeValidated(call, listener, index);
                        } catch (IOException e) {
                            throw new UncheckedIOException(e);
                        }
                    }, toolExecutor));
                }
                List<AgentTools.ToolResult> results = new ArrayList<>();
                boolean clientGone = false;
                for (int i = 0; i < futures.size(); i++) {
                    AgentTools.ToolResult toolResult;
                    try {
                        toolResult = futures.get(i).get();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        return;
                    } catch (ExecutionException e) {
                        if (e.getCause() instanceof UncheckedIOException uio) {
                            throw uio; // client disconnected mid-permission
                        }
                        toolResult = new AgentTools.ToolResult(false,
                                "Инструмент завершился с ошибкой: " + e.getCause().getMessage());
                    }
                    if (toolResult == null) {
                        clientGone = true;
                        break;
                    }
                    results.add(toolResult);

                    ObjectNode resultJson = MAPPER.createObjectNode();
                    resultJson.put("index", i);
                    resultJson.put("tool", calls.get(i).tool());
                    resultJson.put("ok", toolResult.ok());
                    resultJson.put("text", toolResult.text());
                    listener.on("tool_result", resultJson);
                    LOG.info("Agent session " + id + ": tool result [" + calls.get(i).tool()
                            + "] " + (toolResult.ok() ? "ok" : "ERROR") + ": "
                            + abbreviate(toolResult.text(), 200));
                }
                if (clientGone) {
                    return;
                }
                for (int i = 0; i < results.size(); i++) {
                    String note = "[результат инструмента " + calls.get(i).tool()
                            + (results.get(i).ok() ? "" : " — ОШИБКА") + "]: "
                            + results.get(i).text();
                    history.add(new Entry("tool", note));
                    emitHistory("tool", note);
                    ObjectNode toolEntry = MAPPER.createObjectNode();
                    toolEntry.put("tool", calls.get(i).tool());
                    toolEntry.set("args", calls.get(i).args());
                    toolEntry.put("ok", results.get(i).ok());
                    toolEntry.put("text", results.get(i).text());
                    emitTranscript("tool", toolEntry);
                }

                // Pause between steps: the in-flight LLM request has already
                // finished, the agent stops before the next one
                if (pauseRequested && !aborted) {
                    listener.on("paused", MAPPER.createObjectNode());
                    synchronized (pauseLock) {
                        while (pauseRequested && !aborted) {
                            try {
                                pauseLock.wait();
                            } catch (InterruptedException e) {
                                Thread.currentThread().interrupt();
                                aborted = true;
                            }
                        }
                    }
                    if (!aborted) {
                        listener.on("resumed", MAPPER.createObjectNode());
                    }
                }
                if (aborted) {
                    emitDone(listener, totalPrompt, totalCompletion, hasUsage,
                            llmCalls, totalToolCalls, true, -1);
                    return;
                }
            }
        } finally {
            // Stale requests must not block future runs
            permissionRegistry.values().forEach(f -> f.complete(false));
        }
    }

    /**
     * One LLM call. With {@code streamTokens} the content tokens and the
     * reasoning tokens are forwarded to the page as they arrive; the internal
     * planning phase passes {@code false} so its raw JSON never leaks into
     * the chat (and never glues to the next reply inside one code fence).
     * Returns {@code null} when the call failed (the error was already
     * emitted to the listener); client disconnects propagate as
     * {@link UncheckedIOException}.
     */
    private LlmClient.ChatResult callLlm(List<LlmClient.Message> messages, Listener listener,
                                         boolean streamTokens)
            throws IOException {
        try {
            return client.chatStream(messages, LlmClient.ChatOptions.defaults(),
                    token -> {
                        if (!streamTokens) {
                            return;
                        }
                        ObjectNode json = MAPPER.createObjectNode();
                        json.put("text", token);
                        try {
                            listener.on("token", json);
                        } catch (IOException e) {
                            throw new UncheckedIOException(e);
                        }
                    },
                    reasoningToken -> {
                        if (!streamTokens) {
                            return;
                        }
                        ObjectNode json = MAPPER.createObjectNode();
                        json.put("text", reasoningToken);
                        try {
                            listener.on("reasoning", json);
                        } catch (IOException e) {
                            throw new UncheckedIOException(e);
                        }
                    },
                    conn -> activeLlmConnection = conn);
        } catch (UncheckedIOException e) {
            throw e;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            if (!aborted) {
                emitError(listener, "Запрос был прерван.");
            }
            return null;
        } catch (Exception e) {
            if (aborted) {
                // Expected: abort() closed the connection mid-read
                LOG.info("Agent session " + id + ": LLM request cancelled by abort");
                return null;
            }
            LOG.log(Level.WARNING, "Agent LLM request failed", e);
            emitError(listener, "Ошибка запроса к LLM: "
                    + (e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName()));
            return null;
        }
    }

    /**
     * Validates and executes one tool call, asking the user for permission
     * first when the tool mutates files. {@code update_plan} is handled
     * internally — it updates the visible plan instead of touching files.
     * Returns {@code null} when the client disconnected while a permission
     * request was open.
     */
    private AgentTools.ToolResult executeValidated(AgentProtocol.ToolCall call,
                                                   Listener listener, int index)
            throws IOException {
        String validationError = AgentProtocol.validate(call);
        if (validationError != null) {
            LOG.info("Agent session " + id + ": tool call rejected by validation: "
                    + validationError);
            return new AgentTools.ToolResult(false, "Вызов отклонён валидацией: " + validationError);
        }

        // Internal progress tool: no files, no permission
        if ("update_plan".equals(call.tool())) {
            applyPlanUpdate(call.args());
            emitPlan(listener);
            return new AgentTools.ToolResult(true, "План обновлён.");
        }

        if (AgentTools.requiresPermission(call.tool())) {
            String requestId = UUID.randomUUID().toString();
            CompletableFuture<Boolean> decision = new CompletableFuture<>();
            permissionRegistry.put(requestId, decision);
            pendingDecisions.add(decision);

            ObjectNode req = MAPPER.createObjectNode();
            req.put("id", requestId);
            req.put("index", index);
            req.put("action", call.tool());
            req.put("file", call.args().path("name").asText(""));
            if (call.args().path("content").isTextual()) {
                req.put("preview", abbreviate(call.args().path("content").asText(), 500));
            }
            listener.on("permission_request", req);

            boolean allowed;
            try {
                allowed = decision.get(PERMISSION_TIMEOUT_MINUTES, TimeUnit.MINUTES);
            } catch (Exception e) {
                allowed = false;
            } finally {
                permissionRegistry.remove(requestId);
                pendingDecisions.remove(decision);
            }

            ObjectNode resolved = MAPPER.createObjectNode();
            resolved.put("id", requestId);
            resolved.put("allowed", allowed);
            listener.on("permission_resolved", resolved);

            ObjectNode permEntry = MAPPER.createObjectNode();
            permEntry.put("action", call.tool());
            permEntry.put("file", call.args().path("name").asText(""));
            permEntry.put("allowed", allowed);
            emitTranscript("permission", permEntry);

            if (!allowed) {
                return new AgentTools.ToolResult(false,
                        "Пользователь запретил операцию " + call.tool() + ".");
            }
        }

        try {
            return tools.execute(call.tool(),
                    call.args().path("name").asText(null),
                    call.args().path("content").asText(null),
                    call.args().path("expression").asText(null));
        } catch (Exception e) {
            LOG.log(Level.WARNING, "Agent session " + id + ": tool " + call.tool() + " failed", e);
            return new AgentTools.ToolResult(false,
                    "Инструмент завершился с ошибкой: " + e.getMessage());
        }
    }

    /**
     * Asks the user whether the run may continue past the tool-call limit.
     * Uses the same permission channel as file operations. Returns
     * {@code null} when the client disconnected, false when denied/timed out.
     */
    private Boolean askContinue(Listener listener) throws IOException {
        String requestId = UUID.randomUUID().toString();
        CompletableFuture<Boolean> decision = new CompletableFuture<>();
        permissionRegistry.put(requestId, decision);
        pendingDecisions.add(decision);
        try {
            ObjectNode req = MAPPER.createObjectNode();
            req.put("requestId", requestId);
            req.put("max", maxToolCalls);
            listener.on("limit_confirm", req);

            boolean allowed;
            try {
                allowed = decision.get(PERMISSION_TIMEOUT_MINUTES, TimeUnit.MINUTES);
            } catch (Exception e) {
                allowed = false;
            }

            ObjectNode resolved = MAPPER.createObjectNode();
            resolved.put("id", requestId);
            resolved.put("allowed", allowed);
            listener.on("permission_resolved", resolved);
            return allowed;
        } finally {
            permissionRegistry.remove(requestId);
            pendingDecisions.remove(decision);
        }
    }

    /** Replaces the plan with the steps supplied by an update_plan call. */
    private void applyPlanUpdate(JsonNode args) {
        List<PlanStep> updated = new ArrayList<>();
        for (JsonNode s : args.path("steps")) {
            String title = abbreviate(s.path("title").asText(""), 200);
            if (title.isEmpty()) {
                continue;
            }
            String status = s.path("status").asText("pending");
            updated.add(new PlanStep(title, status));
            if (updated.size() >= 10) {
                break;
            }
        }
        if (!updated.isEmpty()) {
            plan = updated;
        }
    }

    /** All steps are done when the final answer is given. */
    private void markPlanDone() {
        if (plan.isEmpty()) {
            return;
        }
        plan = plan.stream()
                .map(s -> new PlanStep(s.title(), "done"))
                .toList();
    }

    private void emitPlan(Listener listener) throws IOException {
        emitPlan(listener, false);
    }

    /**
     * Sends the current plan. {@code initial} marks the planning-phase event:
     * only that one may replace a partial bubble on the page (legacy backends
     * streamed the raw planning JSON into it). Later updates must never touch
     * the answer bubble — the end-of-turn event arrives right after the
     * streamed final answer.
     */
    private void emitPlan(Listener listener, boolean initial) throws IOException {
        if (plan.isEmpty()) {
            return;
        }
        ArrayNode steps = MAPPER.createArrayNode();
        for (PlanStep s : plan) {
            steps.addObject().put("title", s.title()).put("status", s.status());
        }
        ObjectNode json = MAPPER.createObjectNode();
        json.set("steps", steps);
        json.put("initial", initial);
        listener.on("plan", json);
        emitTranscript("plan", json);
    }

    /**
     * Builds the message list for the LLM: pinned system prompt + the context
     * window. In "lastN" mode only user/assistant entries are counted; tool
     * records stay with their turn (the window starts at a user message).
     */
    private List<LlmClient.Message> buildMessages() {
        List<Entry> window = historyWindow();
        List<LlmClient.Message> messages = new ArrayList<>(window.size() + 1);
        messages.add(new LlmClient.Message("system", AgentProtocol.systemPrompt()));
        for (Entry e : window) {
            String role = switch (e.role()) {
                case "user", "assistant" -> e.role();
                default -> "user"; // tool results are fed back as user-role notes
            };
            messages.add(new LlmClient.Message(role, e.content()));
        }
        return messages;
    }

    private List<Entry> historyWindow() {
        if ("none".equals(compression)) {
            return history;
        }
        int n = Math.max(1, contextMessages);
        int counted = 0;
        int cut = 0;
        for (int i = history.size() - 1; i >= 0; i--) {
            Entry e = history.get(i);
            if ("user".equals(e.role()) || "assistant".equals(e.role())) {
                counted++;
            }
            if (counted > n) {
                cut = i + 1;
                break;
            }
        }
        // Do not start mid-turn: move the cut to the next user message
        while (cut < history.size() && !"user".equals(history.get(cut).role())) {
            cut++;
        }
        return history.subList(cut, history.size());
    }

    /** Messages currently inside the context window (for the status line). */
    public int keptMessages() {
        return historyWindow().size();
    }

    public int totalMessages() {
        return history.size();
    }

    private void emitDone(Listener listener, int promptTokens, int completionTokens,
                          boolean hasUsage, int llmCalls, int toolCalls, boolean abortedRun,
                          int replyTokens)
            throws IOException {
        ObjectNode stats = MAPPER.createObjectNode();
        if (hasUsage) {
            stats.put("promptTokens", promptTokens);
            stats.put("completionTokens", completionTokens);
            stats.put("totalTokens", promptTokens + completionTokens);
            if (replyTokens >= 0) {
                // Completion tokens of the final answer only
                stats.put("replyTokens", replyTokens);
            }
        }
        stats.put("llmCalls", llmCalls);
        stats.put("toolCalls", toolCalls);
        ObjectNode done = MAPPER.createObjectNode();
        done.set("stats", stats);
        if (hasUsage) {
            // Accumulate the dialog-wide totals and persist them — the
            // counter must survive an application restart
            sessionPromptTokens = Math.max(sessionPromptTokens, 0) + promptTokens;
            sessionCompletionTokens += completionTokens;
            ObjectNode session = MAPPER.createObjectNode();
            session.put("promptTokens", sessionPromptTokens);
            session.put("completionTokens", sessionCompletionTokens);
            session.put("totalTokens", sessionPromptTokens + sessionCompletionTokens);
            done.set("session", session);
            emitTranscript("usage", session);
        }
        ObjectNode context = MAPPER.createObjectNode();
        context.put("keptMessages", keptMessages());
        context.put("totalMessages", totalMessages());
        done.set("context", context);
        done.put("aborted", abortedRun);
        done.put("durationMs", (System.nanoTime() - turnStartNanos) / 1_000_000);
        listener.on("done", done);
        // Persist the turn summary so the status line can be restored
        // together with the dialog after an application restart
        emitTranscript("turnstats", done);
    }

    private static void emitError(Listener listener, String message) throws IOException {
        ObjectNode json = MAPPER.createObjectNode();
        json.put("error", message);
        listener.on("error", json);
    }

    private static String abbreviate(String text, int maxLength) {
        if (text == null) {
            return "";
        }
        String oneLine = text.replaceAll("\\s+", " ").trim();
        return oneLine.length() <= maxLength ? oneLine : oneLine.substring(0, maxLength) + "…";
    }
}
