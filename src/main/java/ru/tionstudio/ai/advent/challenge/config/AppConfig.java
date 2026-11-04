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

    public String apiKey() {
        return require("llm.api.key");
    }

    public String baseUrl() {
        return get("llm.api.base-url", "https://openrouter.ai/api/v1");
    }

    public String model() {
        return get("llm.model", "openai/gpt-4o-mini");
    }

    public int serverPort() {
        return getInt("server.port", 8080);
    }
}
