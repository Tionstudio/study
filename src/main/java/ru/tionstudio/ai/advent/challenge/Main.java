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
            java.util.List<AppConfig.ModelConfig> modelConfigs = config.modelConfigs();
            AppConfig.ModelConfig defaultGroup = config.defaultModelConfig();
            AppConfig.ModelConfig analysisGroup = config.analysisModelConfig();
            LOG.info("Configuration loaded from " + AppConfig.CONFIG_PATH.toAbsolutePath());
            java.util.Map<String, LlmClient> clients = new java.util.LinkedHashMap<>();
            for (AppConfig.ModelConfig mc : modelConfigs) {
                clients.put(mc.id(), new LlmClient(mc.apiKey(), mc.baseUrl(), mc.model()));
                LOG.info("Model group '" + mc.id() + "': model=" + mc.model()
                        + ", baseUrl=" + mc.baseUrl()
                        + (mc.id().equals(defaultGroup.id()) ? " [default]" : "")
                        + (mc.id().equals(analysisGroup.id()) ? " [analysis]" : ""));
            }
            LlmClient defaultClient = clients.get(defaultGroup.id());
            LlmClient analysisClient = clients.get(analysisGroup.id());
            PresetStore presetStore = new PresetStore(Path.of("settings.db"));
            ConversationStore conversationStore = new ConversationStore(Path.of("settings.db"));
            ConversationManager conversationManager = new ConversationManager(conversationStore, defaultClient);
            ChatServer server = new ChatServer(config.serverPort(), clients, modelConfigs,
                    defaultClient, analysisClient, presetStore, conversationManager);
            server.start();
        } catch (Exception e) {
            LOG.severe("Failed to start: " + e.getMessage());
            System.exit(1);
        }
    }
}
