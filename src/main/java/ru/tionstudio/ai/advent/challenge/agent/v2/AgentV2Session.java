package ru.tionstudio.ai.advent.challenge.agent.v2;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import ru.tionstudio.ai.advent.challenge.agent.AgentProtocol;
import ru.tionstudio.ai.advent.challenge.agent.AgentTools;
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
 *   <li>context compression: "none", "lastN" or "summary" — the window
 *       counts only user/assistant messages, tool records are kept with
 *       their turn; in "summary" mode the older part is replaced by an
 *       LLM-written summary stored separately from the history;</li>
 *   <li>tool calls are validated before execution;</li>
 *   <li>write/append/delete require explicit per-operation user permission;</li>
 *   <li>the number of consecutive tool calls is limited (default 5);</li>
 *   <li>independent calls from one reply run in parallel (virtual threads);</li>
 *   <li>the user may pause processing between agent steps.</li>
 * </ul>
 */
public class AgentV2Session {

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
                           int maxToolCalls, boolean planning,
                           int summaryKeep, int summaryEvery, int slidingMessages,
                           int stickyKeep, int factsMax, int factsConsolidateEvery,
                           boolean toolsEnabled, boolean stickyFacts) {
    }

    private record Entry(String role, String content) {
    }

    /** One plan step as shown in the UI. */
    private record PlanStep(String title, String status) {
    }

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final Logger LOG = Logger.getLogger(AgentV2Session.class.getName());

    /** Permission requests time out after this long and count as denied. */
    private static final long PERMISSION_TIMEOUT_MINUTES = 5;

    /**
     * How many times in a row the model may answer with an unparseable
     * tool call before the turn is ended with an explanatory note.
     */
    private static final int MAX_BROKEN_REPLIES = 3;

    private final String id;
    private final LlmClient client;
    private final AgentTools tools;
    /** Shared registry: permission request id → decision future. */
    private final Map<String, CompletableFuture<Boolean>> permissionRegistry;
    /** Parallel tool execution (virtual threads are cheap to block). */
    private final ExecutorService toolExecutor = Executors.newVirtualThreadPerTaskExecutor();

    private final List<Entry> history = new ArrayList<>();
    private String compression = "summary";
    private int contextMessages = 20;
    private int maxToolCalls = 5;
    private boolean planning = true;
    /**
     * When false, the session behaves as a plain chat: no tools in the
     * system prompt and no tool-call parsing — every reply is a final answer.
     */
    private volatile boolean toolsEnabled = false;
    /** "summary" mode: last N user/assistant messages stay as-is. */
    private int summaryKeep = 5;
    /** "summary" mode: the summary is regenerated every M new messages. */
    private int summaryEvery = 10;
    /** "sliding" mode: hard window — only the last N history entries are kept. */
    private int slidingMessages = 30;
    /** Legacy "sticky" compression: last N user/assistant messages stay verbatim. */
    private int stickyKeep = 20;
    /** Independent sticky-facts switch; enabled by default for every compression mode. */
    private boolean stickyFacts = true;
    /** Max stored facts; the least recently updated are evicted. */
    private int factsMax = 50;
    /** LLM consolidation of the fact list every M counted messages (0 = off). */
    private int factsConsolidateEvery = 0;
    /** countedMessages() value at the last consolidation. */
    private int factsConsolidatedAt = 0;
    /** Counted history position already processed by batch fact extraction. */
    private int factsExtractedAt = 0;
    /** Extracted key-value facts, mirrored in the dialog store. */
    private List<AgentProtocol.Fact> facts = new ArrayList<>();
    /** Persistent storage of the dialog's facts. */
    private final MemoryStore dialogStore;
    /**
     * "summary" mode: compressed older part of the dialog. Stored separately
     * from the history and sent INSTEAD of the entries it covers.
     */
    private String summary = "";
    /** How many user/assistant messages the summary currently covers. */
    private int summaryCoveredCounted;

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

    /** Prompt tokens of the latest LLM call (-1 = none yet this turn). */
    private int lastPromptTokens = -1;
    /** Layer breakdown of the most recently assembled main-chat request. */
    private ObjectNode lastLayerStats = MAPPER.createObjectNode();

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

    public AgentV2Session(String id, LlmClient client, AgentTools tools,
                        MemoryStore dialogStore,
                        Map<String, CompletableFuture<Boolean>> permissionRegistry) {
        this.id = id;
        this.client = client;
        this.tools = tools;
        this.dialogStore = dialogStore;
        this.permissionRegistry = permissionRegistry;
    }

    public String id() {
        return id;
    }

    public String model() {
        return client.model();
    }

    /** Complete inspector snapshot: all three layers plus the latest token estimate. */
    public ObjectNode memorySnapshot() {
        ObjectNode root = MAPPER.createObjectNode();
        root.put("systemPrompt", AgentProtocol.systemPrompt(toolsEnabled));
        root.set("longTerm", dialogStore.dialogAgentConfig(id));
        root.put("profileId", dialogStore.dialogProfileId(id));
        root.put("historyEmpty", dialogStore.loadEntries(id, "history").isEmpty());
        root.set("ruleCandidates", dialogStore.ruleCandidates(true, id));
        root.set("working", dialogStore.working(id));
        ArrayNode factArray = root.putArray("facts");
        for (AgentProtocol.Fact f : dialogStore.listFacts(id)) {
            factArray.addObject().put("category", f.category())
                    .put("key", f.key()).put("value", f.value());
        }
        root.put("summary", summary);
        root.put("summaryCoveredCounted", summaryCoveredCounted);
        root.put("historyMessages", history.size());
        root.put("compression", compression);
        root.put("stickyFacts", stickyFacts);
        root.set("layers", lastLayerStats == null ? MAPPER.createObjectNode() : lastLayerStats);
        return root;
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

    /** Restores the dialog summary stored before a restart ("summary" mode). */
    public void restoreSummary(String text, int coveredCounted) {
        this.summary = text == null ? "" : text;
        this.summaryCoveredCounted = Math.max(0, coveredCounted);
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
        if ("lastN".equals(s.compression()) || "none".equals(s.compression())
                || "summary".equals(s.compression()) || "sliding".equals(s.compression())
                || "sticky".equals(s.compression())) {
            this.compression = s.compression();
        }
        if (s.contextMessages() > 0) {
            this.contextMessages = Math.min(s.contextMessages(), 500);
        }
        if (s.maxToolCalls() > 0) {
            this.maxToolCalls = Math.min(s.maxToolCalls(), 50);
        }
        if (s.summaryKeep() > 0) {
            this.summaryKeep = Math.min(s.summaryKeep(), 100);
        }
        if (s.summaryEvery() > 0) {
            this.summaryEvery = Math.min(s.summaryEvery(), 200);
        }
        if (s.slidingMessages() > 0) {
            this.slidingMessages = Math.min(s.slidingMessages(), 500);
        }
        if (s.stickyKeep() > 0) {
            this.stickyKeep = Math.min(s.stickyKeep(), 500);
        }
        if (s.factsMax() > 0) {
            this.factsMax = Math.min(s.factsMax(), 500);
        }
        if (s.factsConsolidateEvery() >= 0) {
            this.factsConsolidateEvery = Math.min(s.factsConsolidateEvery(), 1000);
        }
        this.planning = s.planning();
        this.toolsEnabled = s.toolsEnabled();
        this.stickyFacts = s.stickyFacts();
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
        LOG.info("Agent v2 session " + id + ": pause requested");
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
        lastPromptTokens = -1;
        history.add(new Entry("user", userMessage));
        emitHistory("user", userMessage);
        emitTranscriptText("user", userMessage);
        plan = new ArrayList<>();

        int consecutiveToolCalls = 0;
        int totalToolCalls = 0;
        int brokenReplies = 0; // consecutive unparseable tool-call attempts
        int totalPrompt = 0, totalCompletion = 0, llmCalls = 0;
        int lastCallCompletion = -1; // completion tokens of the latest LLM call
        boolean hasUsage = false;

        try {
            // ----- Working memory: current task + fresh user message, no full history -----
            if (!aborted) {
                LlmClient.ChatResult wt = updateWorkingMemory(userMessage, listener);
                if (wt != null) {
                    llmCalls++;
                    if (wt.promptTokens() != null) {
                        totalPrompt += wt.promptTokens();
                        lastPromptTokens = wt.promptTokens();
                        hasUsage = true;
                    }
                    if (wt.completionTokens() != null) {
                        totalCompletion += wt.completionTokens();
                        hasUsage = true;
                    }
                }
            }

            // ----- Batch fact extraction before context compression -----
            // Summary mode processes exactly the batch being summarized. Other
            // modes wait until M new uncompressed messages have accumulated.
            if (!aborted) {
                for (LlmClient.ChatResult fx : extractFactsForCompression(listener)) {
                    llmCalls++;
                    if (fx.promptTokens() != null) {
                        totalPrompt += fx.promptTokens();
                        lastPromptTokens = fx.promptTokens();
                        hasUsage = true;
                    }
                    if (fx.completionTokens() != null) {
                        totalCompletion += fx.completionTokens();
                        hasUsage = true;
                    }
                }
            }

            // ----- One-time initial LT adaptation after the first six user messages -----
            if (!aborted) {
                LlmClient.ChatResult bootstrap = bootstrapLongTermIfNeeded(listener);
                if (bootstrap != null) {
                    llmCalls++;
                    if (bootstrap.promptTokens() != null) {
                        totalPrompt += bootstrap.promptTokens();
                        lastPromptTokens = bootstrap.promptTokens();
                        hasUsage = true;
                    }
                    if (bootstrap.completionTokens() != null) {
                        totalCompletion += bootstrap.completionTokens();
                        hasUsage = true;
                    }
                }
            }

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
                        listener, false, "planning");
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
                    lastPromptTokens = planResult.promptTokens();
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

            // ----- Summary maintenance (compression "summary") -----
            if (!aborted) {
                LlmClient.ChatResult sumResult = updateSummaryIfNeeded(listener);
                if (sumResult != null) {
                    llmCalls++;
                    if (sumResult.promptTokens() != null) {
                        totalPrompt += sumResult.promptTokens();
                        lastPromptTokens = sumResult.promptTokens();
                        hasUsage = true;
                    }
                    if (sumResult.completionTokens() != null) {
                        totalCompletion += sumResult.completionTokens();
                        hasUsage = true;
                    }
                }
            }

            // ----- Optional sticky-fact consolidation, independent of compression -----
            if (!aborted && stickyFacts && factsConsolidateEvery > 0
                    && countedMessages() - factsConsolidatedAt >= factsConsolidateEvery) {
                LlmClient.ChatResult cf = consolidateFacts(listener);
                if (cf != null) {
                    llmCalls++;
                    if (cf.promptTokens() != null) {
                        totalPrompt += cf.promptTokens();
                        lastPromptTokens = cf.promptTokens();
                        hasUsage = true;
                    }
                    if (cf.completionTokens() != null) {
                        totalCompletion += cf.completionTokens();
                        hasUsage = true;
                    }
                }
            }

            // ----- Execution loop -----
            while (true) {
                List<LlmClient.Message> assembled = buildMessagesWithBudget(listener);
                if (assembled == null) {
                    return;
                }
                LlmClient.ChatResult result = callLlm(assembled, listener, true, "chat");
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
                    lastPromptTokens = result.promptTokens();
                    hasUsage = true;
                }
                checkContextFill(result.promptTokens(), listener);
                if (result.completionTokens() != null) {
                    totalCompletion += result.completionTokens();
                    lastCallCompletion = result.completionTokens();
                    hasUsage = true;
                }

                String reply = result.reply() == null ? "" : result.reply().trim();
                // Tools disabled (dialog-script plain-chat mode): every reply
                // is a final answer, the tool protocol is not applied at all
                AgentProtocol.ParseResult parsed = toolsEnabled
                        ? AgentProtocol.parseFull(reply)
                        : new AgentProtocol.ParseResult(List.of(), "");
                List<AgentProtocol.ToolCall> calls = parsed.calls();
                if (toolsEnabled && calls.isEmpty() && AgentProtocol.looksLikeToolCall(reply)) {
                    // The model tried to call a tool but produced unparseable
                    // JSON: the raw payload must NOT become a normal chat
                    // answer — flag it as a protocol error and let the model
                    // retry (bounded by MAX_BROKEN_REPLIES)
                    brokenReplies++;
                    history.add(new Entry("assistant", reply));
                    emitHistory("assistant", reply);
                    ObjectNode err = MAPPER.createObjectNode();
                    err.put("text", reply);
                    err.put("attempt", brokenReplies);
                    err.put("maxAttempts", MAX_BROKEN_REPLIES);
                    listener.on("reply_error", err);
                    emitTranscript("reply_error", err);
                    LOG.warning("Agent session " + id + ": unparseable tool call, attempt "
                            + brokenReplies + "/" + MAX_BROKEN_REPLIES + ": "
                            + abbreviate(reply, 200));
                    if (brokenReplies >= MAX_BROKEN_REPLIES) {
                        ObjectNode note = MAPPER.createObjectNode();
                        note.put("text", "Модель " + MAX_BROKEN_REPLIES
                                + " раз(а) подряд прислала повреждённый вызов инструмента"
                                + " — ход завершён. Попробуйте переформулировать запрос.");
                        listener.on("note", note);
                        emitTranscript("note", note);
                        emitDone(listener, totalPrompt, totalCompletion, hasUsage,
                                llmCalls, totalToolCalls, false, lastCallCompletion);
                        return;
                    }
                    String feedback = "[ошибка протокола]: Ваш предыдущий ответ похож на "
                            + "вызов инструмента, но не распознан — JSON повреждён. "
                            + "Повторите вызов СТРОГО одним JSON-объектом "
                            + "{\"tool\": \"<имя>\", \"args\": {...}} без markdown и без "
                            + "пояснений — или дайте финальный ответ обычным текстом.";
                    history.add(new Entry("tool", feedback));
                    emitHistory("tool", feedback);
                    continue;
                }
                if (calls.isEmpty()) {
                    brokenReplies = 0;
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
                brokenReplies = 0;
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
                                         boolean streamTokens) throws IOException {
        return callLlm(messages, listener, streamTokens, streamTokens ? "chat" : "service");
    }

    /** One logged LLM call; the request JSON contains exactly the assembled messages. */
    private LlmClient.ChatResult callLlm(List<LlmClient.Message> messages, Listener listener,
                                         boolean streamTokens, String kind) throws IOException {
        long started = System.nanoTime();
        ObjectNode request = MAPPER.createObjectNode();
        request.put("model", client.model());
        request.put("stream", true);
        if (!streamTokens) request.put("reasoning_effort", "none");
        ArrayNode requestMessages = request.putArray("messages");
        for (LlmClient.Message m : messages) {
            requestMessages.addObject().put("role", m.role()).put("content", m.content());
        }
        LlmClient.ChatResult result = null;
        String failure = "";
        try {
            LlmClient.ChatOptions options = streamTokens
                    ? LlmClient.ChatOptions.defaults()
                    : new LlmClient.ChatOptions("none", "", null, List.of(), "none", null);
            result = client.chatStream(messages, options,
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
            return result;
        } catch (UncheckedIOException e) {
            throw e;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            failure = "Запрос был прерван.";
            if (!aborted) {
                emitError(listener, failure);
            }
            return null;
        } catch (Exception e) {
            failure = e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName();
            if (aborted) {
                LOG.info("Agent session " + id + ": LLM request cancelled by abort");
                return null;
            }
            LOG.log(Level.WARNING, "Agent v2 LLM request failed", e);
            emitError(listener, "Ошибка запроса к LLM: " + failure);
            return null;
        } finally {
            long durationMs = (System.nanoTime() - started) / 1_000_000;
            ObjectNode usage = usageJson(result);
            ObjectNode layers = "chat".equals(kind) && lastLayerStats != null
                    ? lastLayerStats.deepCopy() : MAPPER.createObjectNode();
            try {
                long logId = dialogStore.logLlm(id, kind, request.toString(),
                        result != null ? result.reply() : failure,
                        result != null ? result.reasoning() : "",
                        layers.toString(), usage.toString(), durationMs);
                ObjectNode logRef = MAPPER.createObjectNode();
                logRef.put("logId", logId);
                dialogStore.appendEntry(id, "llmref", logRef.toString());
                ObjectNode logEvent = MAPPER.createObjectNode();
                logEvent.put("id", logId);
                logEvent.put("kind", kind);
                logEvent.set("request", request);
                logEvent.put("response", result != null ? result.reply() : failure);
                logEvent.put("reasoning", result != null ? result.reasoning() : "");
                logEvent.set("layers", layers);
                logEvent.set("usage", usage);
                logEvent.put("durationMs", durationMs);
                listener.on("llm_call", logEvent);
                if ("chat".equals(kind)) {
                    calibrateTokenEstimate(result, layers);
                    listener.on("memory", memorySnapshot());
                }
            } catch (Exception e) {
                LOG.log(Level.WARNING, "Failed to persist/emit v2 LLM log", e);
            }
        }
    }

    private static ObjectNode usageJson(LlmClient.ChatResult result) {
        ObjectNode usage = MAPPER.createObjectNode();
        if (result != null) {
            if (result.promptTokens() != null) usage.put("promptTokens", result.promptTokens());
            if (result.completionTokens() != null) usage.put("completionTokens", result.completionTokens());
            if (result.totalTokens() != null) usage.put("totalTokens", result.totalTokens());
            if (result.reasoningTokens() != null) usage.put("reasoningTokens", result.reasoningTokens());
            if (result.finishReason() != null) usage.put("finishReason", result.finishReason());
        }
        return usage;
    }

    private void calibrateTokenEstimate(LlmClient.ChatResult result, ObjectNode layers) {
        if (result == null || result.promptTokens() == null) {
            return;
        }
        int estimated = layers.path("totalEstTokens").asInt(0);
        if (estimated <= 0 || result.promptTokens() <= 0) {
            return;
        }
        double factor = Math.max(0.3, Math.min(3.0, (double) result.promptTokens() / estimated));
        dialogStore.setState("token_calibration", Double.toString(factor));
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
        List<LlmClient.Message> messages = new ArrayList<>(window.size() + 4);
        String system = AgentProtocol.systemPrompt(toolsEnabled);
        String longTerm = "[LONG-TERM — конфиг агента, передаётся целиком]\n"
                + dialogStore.dialogAgentConfig(id).toString();
        ObjectNode working = dialogStore.working(id);
        String workingText = working.isEmpty() ? "" : "[WORKING — активная задача]\n" + working.toString();
        messages.add(new LlmClient.Message("system", system));
        messages.add(new LlmClient.Message("system", longTerm));
        if (!workingText.isEmpty()) {
            messages.add(new LlmClient.Message("system", workingText));
        }
        StringBuilder shortTerm = new StringBuilder();
        if ("summary".equals(compression) && !summary.isBlank()) {
            String block = "Краткое содержание предыдущей части диалога (автоматическое summary):\n" + summary;
            shortTerm.append(block).append('\n');
            messages.add(new LlmClient.Message("user", block));
        }
        if (stickyFacts && !facts.isEmpty()) {
            String block = factsBlock();
            shortTerm.append(block).append('\n');
            messages.add(new LlmClient.Message("user", block));
        }
        String currentUser = "";
        for (Entry e : window) {
            String role = switch (e.role()) {
                case "user", "assistant" -> e.role();
                default -> "user";
            };
            if ("user".equals(role)) {
                currentUser = e.content();
            }
            shortTerm.append('[').append(e.role()).append("] ").append(e.content()).append('\n');
            messages.add(new LlmClient.Message(role, e.content()));
        }
        double calibration = 1.0;
        try {
            calibration = Double.parseDouble(dialogStore.state("token_calibration"));
        } catch (Exception ignored) {
            // first request or no calibration yet
        }
        lastLayerStats = MemoryAssembler.stats(system, longTerm, workingText,
                shortTerm.toString(), currentUser, calibration);
        return normalizeRoles(messages);
    }

    /** Fits only ST into the model window; LT and WT are never truncated. */
    private List<LlmClient.Message> buildMessagesWithBudget(Listener listener) throws IOException {
        List<LlmClient.Message> messages = buildMessages();
        if (contextWindow <= 0) {
            return messages;
        }
        boolean reduced = false;
        while (lastLayerStats.path("totalEstTokens").asInt(0) > contextWindow
                && history.size() > 1 && slidingMessages > 1) {
            if (!"sliding".equals(compression)) {
                slidingMessages = history.size();
            }
            compression = "sliding";
            slidingMessages = Math.max(1, slidingMessages / 2);
            reduced = true;
            messages = buildMessages();
        }
        if (lastLayerStats.path("totalEstTokens").asInt(0) > contextWindow) {
            emitError(listener, "Запрос не помещается в контекстное окно даже после сжатия short-term. "
                    + "Long-term и working не обрезаны. Упростите конфиг агента или завершите текущую задачу.");
            return null;
        }
        if (reduced) {
            ObjectNode note = MAPPER.createObjectNode();
            note.put("text", "Контекст переполнен: short-term автоматически уменьшен до "
                    + slidingMessages + " последних записей. Long-term и working сохранены целиком.");
            listener.on("note", note);
            emitTranscript("note", note);
        }
        listener.on("memory", memorySnapshot());
        return messages;
    }

    /**
     * LM Studio-style chat templates require strict user/assistant
     * alternation and a user message right after the system one. Compression
     * breaks this naturally: the summary / facts block is a user message that
     * may be followed by a user message from the window, a sliding window may
     * start mid-turn at an assistant reply, and several tool notes in a row
     * are all mapped to the user role. This pass merges consecutive same-role
     * messages into one and, if the dialogue would start with an assistant
     * message, prepends a neutral user marker.
     */
    private static List<LlmClient.Message> normalizeRoles(List<LlmClient.Message> messages) {
        List<LlmClient.Message> out = new ArrayList<>(messages.size());
        for (LlmClient.Message m : messages) {
            if (!"system".equals(m.role()) && !out.isEmpty()
                    && out.get(out.size() - 1).role().equals(m.role())) {
                LlmClient.Message last = out.get(out.size() - 1);
                out.set(out.size() - 1, new LlmClient.Message(
                        last.role(), last.content() + "\n\n" + m.content()));
            } else {
                out.add(m);
            }
        }
        for (int i = 0; i < out.size(); i++) {
            if (!"system".equals(out.get(i).role())) {
                if (!"user".equals(out.get(i).role())) {
                    out.add(i, new LlmClient.Message("user", "(продолжение диалога)"));
                }
                break;
            }
        }
        return out;
    }

    private List<Entry> historyWindow() {
        if ("none".equals(compression)) {
            return history;
        }
        if ("sliding".equals(compression)) {
            // Hard sliding window: the last N history entries regardless of
            // role — tool records count too, and the window may start
            // mid-turn. Everything older is dropped entirely.
            int n = Math.max(1, slidingMessages);
            return history.subList(Math.max(0, history.size() - n), history.size());
        }
        if ("summary".equals(compression)) {
            // Everything the summary covers is replaced by the summary text
            // (injected in buildMessages); the rest of the history goes as-is
            return history.subList(
                    cutIndexForCounted(Math.min(summaryCoveredCounted, countedMessages())),
                    history.size());
        }
        // "lastN" and "sticky": last N user/assistant messages verbatim;
        // tool records stay with their turn (the window starts at a user message)
        int n = Math.max(1, "sticky".equals(compression) ? stickyKeep : contextMessages);
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

    /** Number of user/assistant entries in the history (tool records excluded). */
    private int countedMessages() {
        int c = 0;
        for (Entry e : history) {
            if ("user".equals(e.role()) || "assistant".equals(e.role())) {
                c++;
            }
        }
        return c;
    }

    /**
     * History index of the (covered + 1)-th user/assistant message — the
     * start of the window when {@code covered} counted messages are
     * replaced by the summary.
     */
    private int cutIndexForCounted(int covered) {
        if (covered <= 0) {
            return 0;
        }
        int c = 0;
        for (int i = 0; i < history.size(); i++) {
            Entry e = history.get(i);
            if ("user".equals(e.role()) || "assistant".equals(e.role())) {
                c++;
                if (c > covered) {
                    return i;
                }
            }
        }
        return history.size();
    }

    /** Max chars of one history entry included in a summary-update prompt. */
    private static final int SUMMARY_ENTRY_MAX_CHARS = 800;

    /**
     * Summary maintenance for the "summary" compression mode. When the
     * backlog of counted messages not yet covered by the summary reaches
     * {@code summaryEvery}, the LLM rewrites the summary, extending its
     * coverage by ONE batch of {@code summaryEvery} messages (bounded steps
     * keep the regeneration prompt inside the model's context window; a huge
     * backlog is caught up over several turns). If the batch call fails, one
     * retry with a 2-message batch is attempted; a persistent failure is
     * reported to the chat. Returns the call's result so its tokens can be
     * included in the turn statistics (null when no update was needed).
     */
    private LlmClient.ChatResult updateSummaryIfNeeded(Listener listener) throws IOException {
        if (!"summary".equals(compression)) {
            return null;
        }
        int countedTotal = countedMessages();
        int every = Math.max(1, summaryEvery);
        int target = Math.max(0, countedTotal - Math.max(1, summaryKeep));
        if (target - summaryCoveredCounted < every) {
            return null;
        }
        int batchTarget = Math.min(target, summaryCoveredCounted + every);
        LlmClient.ChatResult result = tryUpdateSummary(batchTarget, listener);
        if (result == null && !aborted && batchTarget - summaryCoveredCounted > 2) {
            // Possibly a few huge tool outputs blew the context window —
            // retry with a minimal batch
            LOG.info("Agent session " + id + ": summary update with batch of "
                    + (batchTarget - summaryCoveredCounted) + " failed, retrying with batch of 2");
            result = tryUpdateSummary(summaryCoveredCounted + 2, listener);
        }
        if (result == null && !aborted) {
            ObjectNode note = MAPPER.createObjectNode();
            note.put("text", "Не удалось обновить summary: вызов LLM завершился ошибкой"
                    + " (возможно, промпт суммаризации не помещается в контекстное окно"
                    + " модели). Диалог продолжается со старым summary.");
            listener.on("note", note);
            emitTranscript("note", note);
        }
        return result;
    }

    /**
     * One summary regeneration attempt covering the history up to
     * {@code newCovered} counted messages. Long entries (tool outputs) are
     * truncated so the prompt stays compact. On success the summary is
     * stored, pushed to the page and persisted; returns null on failure.
     */
    private LlmClient.ChatResult tryUpdateSummary(int newCovered, Listener listener)
            throws IOException {
        StringBuilder prompt = new StringBuilder();
        prompt.append("Ты сжимаешь историю диалога, чтобы она помещалась в контекстное окно.\n")
                .append("Ниже дано прежнее краткое содержание диалога (может отсутствовать) ")
                .append("и новые сообщения, которые нужно в него добавить.\n\n")
                .append("Составь обновлённое краткое содержание: сохрани все важные факты, ")
                .append("решения, имена файлов, числа, промежуточные результаты и намерения ")
                .append("пользователя, нужные для продолжения работы. Пиши кратко, по-русски. ")
                .append("Ответь ТОЛЬКО обновлённым содержанием, без пояснений.\n\n");
        if (!summary.isBlank()) {
            prompt.append("Прежнее краткое содержание:\n").append(summary).append("\n\n");
        }
        prompt.append("Новые сообщения:\n");
        int from = cutIndexForCounted(summaryCoveredCounted);
        int to = cutIndexForCounted(newCovered);
        for (int i = from; i < to; i++) {
            Entry e = history.get(i);
            prompt.append('[').append(e.role()).append("]: ")
                    .append(abbreviate(e.content(), SUMMARY_ENTRY_MAX_CHARS)).append("\n");
        }
        LlmClient.ChatResult result = callLlm(
                List.of(new LlmClient.Message("user", prompt.toString())), listener, false, "summary");
        if (result == null || result.reply() == null || result.reply().isBlank()) {
            LOG.warning("Agent session " + id + ": summary update failed (batch to "
                    + newCovered + " messages), keeping the previous summary");
            return null;
        }
        summary = result.reply().trim();
        summaryCoveredCounted = newCovered;
        ObjectNode json = MAPPER.createObjectNode();
        json.put("text", summary);
        json.put("coveredCounted", summaryCoveredCounted);
        // Push to the page (memory panel) and persist for restart restore
        listener.on("summary", json);
        emitTranscript("summary", json);
        LOG.info("Agent session " + id + ": summary updated, covers " + summaryCoveredCounted
                + " of " + countedMessages() + " messages (" + summary.length() + " chars)");
        return result;
    }

    /** Messages currently inside the context window (for the status line). */
    public int keptMessages() {
        return historyWindow().size();
    }

    public int totalMessages() {
        return history.size();
    }

    /** Restores the fact list stored before a restart ("sticky" mode). */
    public void restoreFacts(List<AgentProtocol.Fact> stored) {
        this.facts = new ArrayList<>(stored);
    }

    /** Max chars of one message handed to the fact-extraction call. */
    private static final int FACT_EXCERPT_MAX_CHARS = 1500;

    /** The fact block injected into the context in "sticky" mode. */
    private String factsBlock() {
        StringBuilder sb = new StringBuilder();
        sb.append("Известные факты о пользователе и задаче (долговременная память, ")
          .append("обновляется автоматически — учитывай их в ответах):\n");
        for (AgentProtocol.Fact f : facts) {
            sb.append("- [").append(f.category()).append("] ")
              .append(f.key()).append(": ").append(f.value()).append("\n");
        }
        return sb.toString().trim();
    }

    /** Pushes the current fact list to the page ("sticky" mode memory panel). */
    private void emitFacts(Listener listener) throws IOException {
        ArrayNode arr = MAPPER.createArrayNode();
        for (AgentProtocol.Fact f : facts) {
            arr.addObject().put("category", f.category())
                    .put("key", f.key()).put("value", f.value());
        }
        ObjectNode json = MAPPER.createObjectNode();
        json.set("facts", arr);
        listener.on("facts", json);
        // Journal the snapshot: mid-dialog checkpoints restore the fact list
        // from the last "facts" entry at their boundary
        emitTranscript("facts", json);
    }

    /**
     * Updates WT incrementally. The model returns only changed fields instead of
     * regenerating the whole task JSON on every user message.
     */
    private LlmClient.ChatResult updateWorkingMemory(String userMessage, Listener listener)
            throws IOException {
        ObjectNode current = dialogStore.working(id);
        String prompt = """
                Обнови рабочую память активной задачи. Вход содержит только прежний JSON
                и новое сообщение пользователя — не выдумывай отсутствующие сведения.
                WT хранит ТОЛЬКО текущую задачу: её формулировку, критерии приёмки,
                ограничения, план, статус, результаты и артефакты. Личные факты пользователя,
                не относящиеся к задаче, не добавляй. Сохраняй ручные правки, пока
                пользователь явно их не меняет.

                Каждый содержательный запрос или вопрос пользователя — это новая активная
                задача, даже если для ответа нужны только сведения из памяти. Обнови task,
                при необходимости acceptanceCriteria и status. Уточнение обновляет текущую
                задачу. Только приветствие, благодарность или small talk без новой задачи
                не меняют WT: верни {"changed":false,"operations":[]}.
                В остальных случаях верни только изменения:
                {"changed":true,"operations":[{"op":"set|append|remove",
                "field":"имя поля","value":значение}]}.
                Поля: task, acceptanceCriteria, constraints, plan, status, results,
                artifacts, notes. task/status/notes — строки, остальные — массивы.
                set заменяет поле, append добавляет элементы массива, remove удаляет
                элементы массива по точному значению. Не возвращай неизменённые поля.
                Ответь СТРОГО одним JSON-объектом без markdown.

                Прежняя рабочая память:
                %s

                Новое сообщение пользователя:
                %s
                """.formatted(current.toString(), userMessage);
        LlmClient.ChatResult result = callLlm(List.of(new LlmClient.Message("user", prompt)),
                listener, false, "working_update");
        if (result == null || result.reply() == null || result.reply().isBlank()) {
            return result;
        }
        try {
            String text = result.reply().trim()
                    .replaceFirst("^```(?:json)?\\s*", "").replaceFirst("```$", "").trim();
            int start = text.indexOf('{');
            int end = text.lastIndexOf('}');
            if (start >= 0 && end > start) {
                text = text.substring(start, end + 1);
            }
            JsonNode parsed = MAPPER.readTree(text);
            ObjectNode updated = applyWorkingPatch(current, parsed);
            if (updated != null) {
                dialogStore.saveWorking(id, updated);
                listener.on("working", updated);
                listener.on("memory", memorySnapshot());
                emitTranscript("working", updated);
            }
        } catch (Exception e) {
            LOG.warning("Agent session " + id + ": invalid working-memory patch: "
                    + abbreviate(result.reply(), 200) + " (" + e.getMessage() + ")");
        }
        return result;
    }

    /** Applies a validated patch; null means that the task did not change. */
    private static ObjectNode applyWorkingPatch(ObjectNode current, JsonNode parsed) {
        if (parsed == null || !parsed.isObject()) {
            throw new IllegalArgumentException("WT patch must be an object");
        }
        if (parsed.has("changed") && !parsed.path("changed").asBoolean()) {
            return null;
        }
        JsonNode operations = parsed.get("operations");
        if (operations == null || !operations.isArray()) {
            throw new IllegalArgumentException("WT patch must contain an operations array");
        }
        ObjectNode updated = current.deepCopy();
        for (JsonNode operation : operations) {
            String op = operation.path("op").asText("set");
            String field = operation.path("field").asText("");
            JsonNode value = operation.get("value");
            if (value == null) {
                throw new IllegalArgumentException("WT operation requires value");
            }
            switch (op) {
                case "set" -> setWorkingField(updated, field, value);
                case "append" -> appendWorkingField(updated, field, value);
                case "remove" -> removeWorkingField(updated, field, value);
                default -> throw new IllegalArgumentException("Unknown WT operation: " + op);
            }
        }
        return updated.equals(current) ? null : updated;
    }

    private static void requireWorkingField(String field) {
        if (!List.of("task", "acceptanceCriteria", "constraints", "plan", "status",
                "results", "artifacts", "notes").contains(field)) {
            throw new IllegalArgumentException("Unknown WT field: " + field);
        }
    }

    private static void requireWorkingArray(String field) {
        requireWorkingField(field);
        if (!List.of("acceptanceCriteria", "constraints", "plan", "results", "artifacts")
                .contains(field)) {
            throw new IllegalArgumentException("WT field must be an array: " + field);
        }
    }

    private static void setWorkingField(ObjectNode target, String field, JsonNode value) {
        requireWorkingField(field);
        boolean array = List.of("acceptanceCriteria", "constraints", "plan", "results",
                "artifacts").contains(field);
        if ((array && !value.isArray()) || (!array && !value.isTextual())) {
            throw new IllegalArgumentException("Invalid WT value type for " + field);
        }
        target.set(field, value.deepCopy());
    }

    private static void appendWorkingField(ObjectNode target, String field, JsonNode value) {
        requireWorkingArray(field);
        ArrayNode array = target.withArray(field);
        if (value.isArray()) {
            for (JsonNode item : value) appendUnique(array, item);
        } else {
            appendUnique(array, value);
        }
    }

    private static void appendUnique(ArrayNode target, JsonNode value) {
        for (JsonNode existing : target) {
            if (existing.equals(value)) return;
        }
        target.add(value.deepCopy());
    }

    private static void removeWorkingField(ObjectNode target, String field, JsonNode value) {
        requireWorkingArray(field);
        ArrayNode array = target.withArray(field);
        if (value.isArray()) {
            for (JsonNode item : value) removeValue(array, item);
        } else {
            removeValue(array, value);
        }
    }

    private static void removeValue(ArrayNode target, JsonNode value) {
        for (int i = target.size() - 1; i >= 0; i--) {
            if (target.get(i).equals(value)) target.remove(i);
        }
    }

    /** Selects one batch without ever extracting facts on every individual request. */
    private List<LlmClient.ChatResult> extractFactsForCompression(Listener listener) throws IOException {
        if (!stickyFacts) return List.of();
        int total = countedMessages();
        int start;
        int end;
        if ("summary".equals(compression)) {
            int target = Math.max(0, total - Math.max(1, summaryKeep));
            int every = Math.max(1, summaryEvery);
            if (target - summaryCoveredCounted < every) return List.of();
            start = summaryCoveredCounted;
            end = Math.min(target, summaryCoveredCounted + every);
        } else {
            int batch = Math.max(1, stickyKeep);
            if (total - factsExtractedAt < batch) return List.of();
            end = total;
            start = Math.max(factsExtractedAt, total - batch);
        }
        StringBuilder text = new StringBuilder();
        int position = 0;
        for (Entry entry : history) {
            if (!"user".equals(entry.role()) && !"assistant".equals(entry.role())) continue;
            if (position >= start && position < end) {
                text.append('[').append(entry.role()).append("]: ")
                        .append(abbreviate(entry.content(), FACT_EXCERPT_MAX_CHARS)).append('\n');
            }
            position++;
        }
        factsExtractedAt = end;
        if (text.isEmpty()) return List.of();
        return extractFacts(text.toString(), false, listener);
    }

    /**
     * Independently enabled fact extraction. User preferences additionally pass
     * through a separate, context-free LT-candidate call. All calls are returned
     * so their usage is included in turn statistics.
     */
    private List<LlmClient.ChatResult> extractFacts(String text, boolean fromAssistant,
                                                    Listener listener) throws IOException {
        if (!stickyFacts || text == null || text.isBlank()) {
            return List.of();
        }
        String excerpt = abbreviate(text, FACT_EXCERPT_MAX_CHARS);
        LlmClient.ChatResult result = callLlm(
                List.of(new LlmClient.Message("user",
                        AgentProtocol.factExtractionPrompt(excerpt, fromAssistant))),
                listener, false, "extract_st");
        if (result == null) {
            return List.of(); // error already reported; extraction is best-effort
        }
        List<LlmClient.ChatResult> results = new ArrayList<>();
        results.add(result);
        List<AgentProtocol.Fact> extracted = AgentProtocol.parseFacts(result.reply());
        if (!extracted.isEmpty()) {
            facts = dialogStore.upsertFacts(id, extracted, factsMax);
            emitFacts(listener);
            LOG.info("Agent session " + id + ": " + extracted.size() + " fact(s) extracted from "
                    + (fromAssistant ? "assistant reply" : "user message")
                    + ", total stored: " + facts.size());
            if (!fromAssistant && !aborted) {
                LlmClient.ChatResult candidates = detectRuleCandidates(extracted, listener);
                if (candidates != null) results.add(candidates);
            }
        }
        return results;
    }

    /** One-time initial adaptation; never overwrites the configuration directly. */
    private LlmClient.ChatResult bootstrapLongTermIfNeeded(Listener listener) throws IOException {
        if ("true".equals(dialogStore.state("lt_bootstrap_completed:" + id))) return null;
        StringBuilder messages = new StringBuilder();
        int userMessages = 0;
        for (Entry entry : history) {
            if (!"user".equals(entry.role())) continue;
            userMessages++;
            if (userMessages <= 6) messages.append(abbreviate(entry.content(), 1200)).append("\n---\n");
        }
        if (userMessages < 6) return null;
        String prompt = """
                Предложи первичную адаптацию постоянной конфигурации агента по первым
                сообщениям пользователя и извлечённым фактам. Учитывай только явно выраженные
                устойчивые предпочтения взаимодействия. Не включай данные задачи и не выдумывай.
                Не дублируй существующие правила. Можно добавить правило или уточнить одно из
                встроенных default-* правил. Для update обязательно скопируй его текущий текст
                в oldText. Ответь строго JSON без markdown:
                {"candidates":[{"operation":"add|update","key":"stable_snake_case",
                "ruleId":"default-id для update","oldText":"текущий текст для update",
                "text":"предлагаемое правило","sourceFact":"источник"}]}
                Если адаптация не нужна, верни {"candidates":[]}.

                Первые сообщения: %s
                Факты: %s
                Текущий LT: %s
                """.formatted(messages.toString(), facts.toString(), dialogStore.dialogAgentConfig(id).toString());
        LlmClient.ChatResult result = callLlm(List.of(new LlmClient.Message("user", prompt)),
                listener, false, "lt_bootstrap");
        if (result == null || result.reply() == null || result.reply().isBlank()) return result;
        try {
            String text = result.reply().trim()
                    .replaceFirst("^```(?:json)?\\s*", "").replaceFirst("```$", "").trim();
            JsonNode parsed = MAPPER.readTree(text);
            ObjectNode config = dialogStore.dialogAgentConfig(id);
            for (JsonNode candidate : parsed.path("candidates")) {
                String operation = candidate.path("operation").asText("add");
                String target = candidate.path("ruleId").asText("").trim();
                String rule = candidate.path("text").asText("").trim();
                if (rule.isBlank() || !List.of("add", "update").contains(operation)) continue;
                if ("update".equals(operation)) {
                    if (!target.startsWith("default-")) continue;
                    String expected = candidate.path("oldText").asText("");
                    boolean unchanged = false;
                    for (JsonNode existing : config.path("rules")) {
                        if (target.equals(existing.path("id").asText())
                                && expected.equals(existing.path("text").asText())) unchanged = true;
                    }
                    if (!unchanged) continue;
                }
                String rawKey = "update".equals(operation) ? target : candidate.path("key").asText("preference");
                String key = ("bootstrap-" + rawKey).toLowerCase().replaceAll("[^a-z0-9_]", "_");
                dialogStore.observeRuleCandidate(key, rule, candidate.path("sourceFact").asText(""), id,
                        operation, target, candidate.path("oldText").asText(""), "bootstrap");
            }
            dialogStore.setState("lt_bootstrap_completed:" + id, "true");
            listener.on("memory", memorySnapshot());
        } catch (Exception e) {
            LOG.warning("Agent session " + id + ": invalid LT bootstrap: "
                    + abbreviate(result.reply(), 200) + " (" + e.getMessage() + ")");
        }
        return result;
    }

    /** Turns only durable interaction preferences into pending LT rule candidates. */
    private LlmClient.ChatResult detectRuleCandidates(List<AgentProtocol.Fact> extracted,
                                                       Listener listener) throws IOException {
        ArrayNode factsJson = MAPPER.createArrayNode();
        for (AgentProtocol.Fact f : extracted) {
            factsJson.addObject().put("key", f.key()).put("value", f.value());
        }
        String prompt = """
                Найди среди новых фактов только УСТОЙЧИВЫЕ ПРЕДПОЧТЕНИЯ взаимодействия
                с агентом: язык, стиль, детализацию, формат ответов и постоянные ограничения.
                Не переноси имена, адреса, увлечения, разовые просьбы и данные задачи.
                Сравни кандидатов с существующими правилами: дубликаты и противоречащие
                более позднему предпочтению правила не добавляй. Сформулируй самостоятельное
                правило на русском языке. Ответь строго JSON без markdown:
                {"candidates":[{"key":"stable_snake_case_key","text":"правило","sourceFact":"ключ факта"}]}
                Если подходящих фактов нет, верни {"candidates":[]}.

                Новые факты: %s
                Существующие правила LT: %s
                """.formatted(factsJson.toString(), dialogStore.dialogAgentConfig(id).toString());
        LlmClient.ChatResult result = callLlm(List.of(new LlmClient.Message("user", prompt)),
                listener, false, "lt_candidates");
        if (result == null || result.reply() == null || result.reply().isBlank()) return result;
        try {
            String text = result.reply().trim()
                    .replaceFirst("^```(?:json)?\\s*", "").replaceFirst("```$", "").trim();
            JsonNode parsed = MAPPER.readTree(text);
            for (JsonNode candidate : parsed.path("candidates")) {
                String key = candidate.path("key").asText("").trim().toLowerCase()
                        .replaceAll("[^a-z0-9_]", "_");
                String rule = candidate.path("text").asText("").trim();
                String source = candidate.path("sourceFact").asText("").trim();
                if (!key.isBlank() && !rule.isBlank()) {
                    dialogStore.observeRuleCandidate(key, rule, source, id,
                            "add", "", "", "preferences");
                }
            }
            listener.on("memory", memorySnapshot());
        } catch (Exception e) {
            LOG.warning("Agent session " + id + ": invalid LT candidates: "
                    + abbreviate(result.reply(), 200) + " (" + e.getMessage() + ")");
        }
        return result;
    }

    /**
     * Periodic LLM consolidation of the fact list ("sticky" mode, when
     * {@code factsConsolidateEvery > 0}): merges duplicates/synonyms and drops
     * contradicted facts. Best-effort: on failure the old list is kept.
     */
    private LlmClient.ChatResult consolidateFacts(Listener listener) throws IOException {
        factsConsolidatedAt = countedMessages();
        if (facts.isEmpty()) {
            return null;
        }
        ArrayNode arr = MAPPER.createArrayNode();
        for (AgentProtocol.Fact f : facts) {
            arr.addObject().put("category", f.category())
                    .put("key", f.key()).put("value", f.value());
        }
        LlmClient.ChatResult result = callLlm(
                List.of(new LlmClient.Message("user",
                        AgentProtocol.factConsolidationPrompt(arr.toString()))),
                listener, false, "facts_consolidate");
        if (result == null || result.reply() == null || result.reply().isBlank()) {
            LOG.warning("Agent session " + id + ": fact consolidation failed, keeping the list");
            return result;
        }
        List<AgentProtocol.Fact> merged = AgentProtocol.parseFacts(result.reply());
        if (!merged.isEmpty()) {
            facts = dialogStore.replaceFacts(id, merged);
            emitFacts(listener);
            LOG.info("Agent session " + id + ": facts consolidated to " + facts.size());
        }
        return result;
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
        if ("summary".equals(compression) && summaryCoveredCounted > 0) {
            // How many user/assistant messages the summary currently
            // replaces — shown in the turn status line
            context.put("summaryCovered", summaryCoveredCounted);
        }
        if (contextWindow > 0 && lastPromptTokens >= 0) {
            // Context-window fill of the latest LLM call — the page shows
            // it as a chip with yellow/red highlight near the limit
            context.put("windowPct", (int) (lastPromptTokens * 100L / contextWindow));
            context.put("windowTokens", lastPromptTokens);
            context.put("windowSize", contextWindow);
        }
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
