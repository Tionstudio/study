package ru.tionstudio.ai.advent.challenge.conversation;

import com.fasterxml.jackson.databind.JsonNode;
import ru.tionstudio.ai.advent.challenge.conversation.ConversationStore.StoredMessage;
import ru.tionstudio.ai.advent.challenge.llm.LlmClient;
import ru.tionstudio.ai.advent.challenge.llm.LlmClient.ChatOptions;
import ru.tionstudio.ai.advent.challenge.llm.LlmClient.Message;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

/**
 * Builds the {@code messages} array for an LLM request in context mode:
 * keeps the conversation history server-side, fits it into the configured
 * context limit and compresses the overflow according to the chosen strategy
 * (truncate / summarize / hybrid).
 */
public class ConversationManager {

    private static final Logger LOG = Logger.getLogger(ConversationManager.class.getName());

    /**
     * In "hybrid" mode at least this many of the newest messages are always
     * sent verbatim, even if they overflow the limit on their own.
     */
    private static final int HYBRID_VERBATIM_MESSAGES = 6;

    /** Rough estimate for token counting without a tokenizer (~3 chars/token for RU). */
    private static int estimateTokens(String text) {
        return Math.max(1, (text.length() + 2) / 3 + 4);
    }

    /**
     * Context-mode settings from the web UI.
     *
     * @param unit     "messages" | "tokens"
     * @param strategy "truncate" | "summarize" | "hybrid"
     */
    public record ContextSettings(boolean enabled, String unit, int limit, String strategy) {

        public static ContextSettings disabled() {
            return new ContextSettings(false, "tokens", 8000, "summarize");
        }

        public static ContextSettings parse(JsonNode json) {
            if (!json.path("useContext").asBoolean(false)) {
                return disabled();
            }
            // UI exposes only the token limit; "messages" stays for API callers
            String unitRaw = json.path("contextUnit").asText("tokens");
            String unit = "messages".equals(unitRaw) ? "messages" : "tokens";
            int defaultLimit = "tokens".equals(unit) ? 8000 : 20;
            int maxLimit = "tokens".equals(unit) ? 1_000_000 : 1000;
            JsonNode limitNode = json.path("contextLimit");
            int limit = limitNode.isInt() ? limitNode.asInt() : defaultLimit;
            limit = Math.max(1, Math.min(limit, maxLimit));
            // UI always uses summarization; truncate/hybrid stay for API callers
            String strategyRaw = json.path("contextStrategy").asText("summarize");
            String strategy = switch (strategyRaw) {
                case "truncate", "hybrid" -> strategyRaw;
                default -> "summarize";
            };
            return new ContextSettings(true, unit, limit, strategy);
        }
    }

    /**
     * The request ready to be sent to the LLM.
     *
     * @param messages     history window (+ optional summary system message) + new user message
     * @param compressed   true when part of the history was dropped or summarized
     * @param keptMessages how many stored messages are sent verbatim
     * @param totalMessages total messages stored in the conversation
     * @param summaryUsed  true when a summary system message is included
     * @param summaryChars length of the summary text (0 when no summary)
     * @param auxTokens    tokens spent on auxiliary calls (summarization), if reported
     * @param fillPercent  how full the context limit was on this request (0–100)
     */
    public record Prepared(List<Message> messages, String conversationId,
                           boolean compressed, int keptMessages, int totalMessages,
                           boolean summaryUsed, int summaryChars, int auxTokens,
                           int fillPercent) {
    }

    private final ConversationStore store;
    private final LlmClient llmClient;

    public ConversationManager(ConversationStore store, LlmClient llmClient) {
        this.store = store;
        this.llmClient = llmClient;
    }

    /**
     * Registers the user message in the conversation and builds the LLM request
     * honouring the context limit and compression strategy.
     */
    public Prepared prepare(String conversationId, String userMessage, ContextSettings settings) {
        if (!settings.enabled()) {
            return new Prepared(List.of(new Message("user", userMessage)),
                    null, false, 1, 1, false, 0, 0, 0);
        }

        String id = (conversationId != null && !conversationId.isBlank() && store.exists(conversationId))
                ? conversationId
                : store.create();
        store.append(id, "user", userMessage);

        ConversationStore.Conversation conv = store.load(id);
        List<StoredMessage> all = conv.messages();
        int size = all.size();
        String summary = conv.summary();
        int summaryUpto = conv.summaryUpto();

        int keepFrom = windowStart(all, settings);
        if ("hybrid".equals(settings.strategy())) {
            // Guarantee the newest messages stay verbatim even past the limit
            keepFrom = Math.min(keepFrom, Math.max(0, size - HYBRID_VERBATIM_MESSAGES));
        }

        // Limit fill estimate: unsummarized part (+ summary) vs the limit,
        // measured before compression — shows how close this request came
        int fillPercent;
        if ("tokens".equals(settings.unit())) {
            int est = estimateTokens(summary);
            for (int i = summaryUpto; i < size; i++) {
                est += estimateTokens(all.get(i).content());
            }
            fillPercent = (int) Math.min(100, Math.round(100.0 * est / settings.limit()));
        } else {
            int unsummarized = size - summaryUpto;
            fillPercent = (int) Math.min(100, Math.round(100.0 * unsummarized / settings.limit()));
        }

        boolean wantsSummary = "summarize".equals(settings.strategy())
                || "hybrid".equals(settings.strategy());
        int auxTokens = 0;
        if (keepFrom > summaryUpto && wantsSummary) {
            try {
                LlmClient.ChatResult sumResult = summarize(summary, all.subList(summaryUpto, keepFrom));
                if (sumResult.reply().isBlank()) {
                    throw new IllegalStateException("LLM returned an empty summary");
                }
                summary = sumResult.reply().trim();
                auxTokens = sumResult.totalTokens() != null ? sumResult.totalTokens() : 0;
                store.updateSummary(id, summary, keepFrom);
            } catch (Exception e) {
                // Compression is best-effort: fall back to silent truncation
                // rather than failing the user's request.
                LOG.warning("Summarization failed, falling back to truncation: " + e.getMessage());
            }
        }

        List<Message> request = new ArrayList<>();
        if (!summary.isBlank()) {
            request.add(new Message("system",
                    "Краткое содержание предыдущей части диалога (это контекст, не упоминай сводку в ответе):\n"
                            + summary));
        }
        for (int i = keepFrom; i < size; i++) {
            request.add(new Message(all.get(i).role(), all.get(i).content()));
        }

        boolean compressed = keepFrom > 0 || !summary.isBlank();
        if (compressed) {
            LOG.info("Conversation " + id + ": context window keeps " + (size - keepFrom)
                    + " of " + size + " messages (strategy=" + settings.strategy()
                    + ", summary=" + (summary.isBlank() ? "no" : summary.length() + " chars") + ")");
        }
        return new Prepared(request, id, compressed, size - keepFrom, size,
                !summary.isBlank(), summary.length(), auxTokens, fillPercent);
    }

    /** Stores the assistant's reply in the conversation (no-op in stateless mode). */
    public void onReply(String conversationId, String reply) {
        if (conversationId != null && reply != null && !reply.isBlank()) {
            store.append(conversationId, "assistant", reply);
        }
    }

    /** Loads a conversation for UI restoration; {@code null} if unknown. */
    public ConversationStore.Conversation load(String id) {
        return (id == null || id.isBlank()) ? null : store.load(id);
    }

    /** Deletes a conversation ("Новый диалог" cleanup). */
    public boolean delete(String id) {
        return id != null && !id.isBlank() && store.delete(id);
    }

    /** Index of the first message kept verbatim; everything before it is out of window. */
    private static int windowStart(List<StoredMessage> all, ContextSettings settings) {
        int size = all.size();
        if ("tokens".equals(settings.unit())) {
            int budget = settings.limit();
            int used = 0;
            int i = size - 1;
            for (; i >= 0; i--) {
                int cost = estimateTokens(all.get(i).content());
                if (used + cost > budget && i < size - 1) {
                    break; // budget exhausted, but the latest message is always kept
                }
                used += cost;
            }
            return Math.max(0, i + 1);
        }
        return Math.max(0, size - settings.limit());
    }

    /** Compresses old messages (merged into the previous summary) via a cheap LLM call. */
    private LlmClient.ChatResult summarize(String previousSummary, List<StoredMessage> chunk)
            throws Exception {
        StringBuilder prompt = new StringBuilder();
        if (previousSummary != null && !previousSummary.isBlank()) {
            prompt.append("Прежняя сводка диалога:\n").append(previousSummary).append("\n\n");
        }
        prompt.append("Сообщения диалога, которые нужно добавить в сводку:\n");
        for (StoredMessage m : chunk) {
            prompt.append("user".equals(m.role()) ? "Пользователь: " : "Ассистент: ")
                    .append(m.content()).append("\n");
        }
        prompt.append("\nСожми всё это в одну краткую сводку (до 150 слов) на языке диалога: "
                + "зафиксируй темы, вопросы пользователя, важные факты, договорённости и выводы, "
                + "нужные для продолжения разговора. Пиши только сводку, без комментариев.");

        ChatOptions options = new ChatOptions("none", "", null, List.of(), "low", null);
        LOG.info("Summarizing " + chunk.size() + " old messages for context compression");
        return llmClient.chatOnce(List.of(new Message("user", prompt.toString())), options);
    }
}
