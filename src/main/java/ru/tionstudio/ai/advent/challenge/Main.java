package ru.tionstudio.ai.advent.challenge;

import ru.tionstudio.ai.advent.challenge.config.AppConfig;
import ru.tionstudio.ai.advent.challenge.conversation.ConversationManager;
import ru.tionstudio.ai.advent.challenge.conversation.ConversationStore;
import ru.tionstudio.ai.advent.challenge.llm.LlmClient;
import ru.tionstudio.ai.advent.challenge.presets.PresetStore;
import ru.tionstudio.ai.advent.challenge.server.ChatServer;

import java.nio.file.Path;
import java.util.logging.Logger;

public class Main {

    static {
        // Single-line log format: timestamp, logger, level, message
        System.setProperty("java.util.logging.SimpleFormatter.format",
                "%1$tF %1$tT %3$s %4$s: %5$s%6$s%n");
    }

    private static final Logger LOG = Logger.getLogger(Main.class.getName());

    public static void main(String[] args) {
        try {
            AppConfig config = AppConfig.load();
            LOG.info("Configuration loaded from " + AppConfig.CONFIG_PATH.toAbsolutePath()
                    + " (model=" + config.model() + ", baseUrl=" + config.baseUrl() + ")");
            PresetStore presetStore = new PresetStore(Path.of("settings.db"));
            ConversationStore conversationStore = new ConversationStore(Path.of("settings.db"));
            LlmClient llmClient = new LlmClient(config);
            ConversationManager conversationManager = new ConversationManager(conversationStore, llmClient);
            ChatServer server = new ChatServer(config.serverPort(), llmClient, presetStore, conversationManager);
            server.start();
        } catch (Exception e) {
            LOG.severe("Failed to start: " + e.getMessage());
            System.exit(1);
        }
    }
}
