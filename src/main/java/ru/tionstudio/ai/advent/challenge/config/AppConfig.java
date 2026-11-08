package ru.tionstudio.ai.advent.challenge.config;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;

/**
 * Loads application configuration from {@code config.properties}
 * located in the working directory (project root).
 *
 * <p>The file is intentionally NOT committed to git (see .gitignore)
 * because it contains the LLM API key. Use {@code config.properties.example}
 * as a template.</p>
 */
public class AppConfig {

    public static final Path CONFIG_PATH = Path.of("config.properties");

    private final Properties props = new Properties();

    private AppConfig() {
    }

    public static AppConfig load() throws IOException {
        AppConfig config = new AppConfig();
        if (!Files.exists(CONFIG_PATH)) {
            throw new IOException("Configuration file not found: " + CONFIG_PATH.toAbsolutePath()
                    + "\nCopy config.properties.example to config.properties and fill in your API key.");
        }
        try (InputStream in = Files.newInputStream(CONFIG_PATH)) {
            config.props.load(in);
        }
        return config;
    }

    public String get(String key, String defaultValue) {
        String value = props.getProperty(key);
        return (value == null || value.isBlank()) ? defaultValue : value.trim();
    }

    public String require(String key) {
        String value = get(key, null);
        if (value == null) {
            throw new IllegalStateException("Missing required property '" + key + "' in " + CONFIG_PATH);
        }
        return value;
    }

    public int getInt(String key, int defaultValue) {
        String value = get(key, null);
        return value == null ? defaultValue : Integer.parseInt(value);
    }

    /**
     * Which LLM backend to use: {@code cloud} (default, remote API with a key)
     * or {@code local-server} (OpenAI-compatible local server such as LM Studio
     * or llama.cpp — key usually not required).
     */
    public String provider() {
        return get("llm.provider", "cloud");
    }

    private boolean isLocalServer() {
        String p = provider();
        return "local-server".equalsIgnoreCase(p) || "local".equalsIgnoreCase(p);
    }

    public String apiKey() {
        if (isLocalServer()) {
            // Local servers normally ignore the key, but the header must exist
            return get("llm.local.key", "not-required");
        }
        return require("llm.api.key");
    }

    public String baseUrl() {
        if (isLocalServer()) {
            return get("llm.local.base-url", "http://localhost:1234/v1");
        }
        return get("llm.api.base-url", "https://openrouter.ai/api/v1");
    }

    public String model() {
        if (isLocalServer()) {
            return require("llm.local.model");
        }
        return get("llm.model", "openai/gpt-4o-mini");
    }

    // ----- Explicit cloud settings (llm.api.* / llm.model), independent of provider -----
    // Used for features that must always run on the cloud model (e.g. result analysis).

    public boolean hasCloudConfig() {
        return get("llm.api.key", null) != null;
    }

    public String cloudApiKey() {
        return require("llm.api.key");
    }

    public String cloudBaseUrl() {
        return get("llm.api.base-url", "https://openrouter.ai/api/v1");
    }

    public String cloudModel() {
        return get("llm.model", "openai/gpt-4o-mini");
    }

    public int serverPort() {
        return getInt("server.port", 8080);
    }
}
