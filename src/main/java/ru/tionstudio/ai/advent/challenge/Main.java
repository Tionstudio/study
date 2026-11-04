package ru.tionstudio.ai.advent.challenge;

import ru.tionstudio.ai.advent.challenge.config.AppConfig;
import ru.tionstudio.ai.advent.challenge.llm.LlmClient;
import ru.tionstudio.ai.advent.challenge.server.ChatServer;

public class Main {
    public static void main(String[] args) {
        try {
            AppConfig config = AppConfig.load();
            LlmClient llmClient = new LlmClient(config);
            ChatServer server = new ChatServer(config.serverPort(), llmClient);
            server.start();
        } catch (Exception e) {
            System.err.println("Failed to start: " + e.getMessage());
            System.exit(1);
        }
    }
}
