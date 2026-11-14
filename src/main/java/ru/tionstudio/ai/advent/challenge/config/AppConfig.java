package ru.tionstudio.ai.advent.challenge.config;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;

/**
 * Loads application configuration from {@code config.properties}
 * located in the working directory (project root).
 *
 * <p>The file is intentionally NOT committed to git (see .gitignore)
 * because it contains the LLM API keys. Use {@code config.properties.example}
 * as a template.</p>
 *
 * <p>Model settings are organized into named groups: {@code llm.models}
 * lists the group ids, and each group has {@code llm.models.<id>.model},
 * {@code llm.models.<id>.api.key} and {@code llm.models.<id>.api.base-url}.
 * {@code llm.default-model} picks the group used for requests by default,
 * {@code llm.analysis.model} the one used for batch-result analysis.</p>
 */
public class AppConfig {

    public static final Path CONFIG_PATH = Path.of("config.properties");

    /** One model settings group: id, model name, API key and base URL.
     * Optional prices are in USD per 1M tokens; null = cost not tracked
     * (e.g. local models). Optional contextWindow — the model's context
     * size in tokens; null/0 = unknown (no fill-level warnings). */
    public record ModelConfig(String id, String model, String apiKey, String baseUrl,
                              Double pricePrompt, Double priceCompletion,
                              Integer contextWindow) {
    }

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
     * All configured model groups, in the order given by {@code llm.models}.
     * Groups missing {@code model} or {@code api.base-url} are skipped.
     * When no groups are configured, falls back to the legacy single-provider
     * keys ({@code llm.provider} / {@code llm.api.*} / {@code llm.local.*}).
     */
    public List<ModelConfig> modelConfigs() {
        List<ModelConfig> list = new ArrayList<>();
        String raw = get("llm.models", "");
        for (String id : Arrays.stream(raw.split(","))
                .map(String::trim).filter(s -> !s.isEmpty()).toList()) {
            String model = get("llm.models." + id + ".model", null);
            String apiKey = get("llm.models." + id + ".api.key", "not-required");
            String baseUrl = get("llm.models." + id + ".api.base-url", null);
            Double pricePrompt = getPrice("llm.models." + id + ".price.prompt");
            Double priceCompletion = getPrice("llm.models." + id + ".price.completion");
            Integer contextWindow = getIntOrNull("llm.models." + id + ".context-window");
            if (model != null && baseUrl != null) {
                list.add(new ModelConfig(id, model, apiKey, baseUrl,
                        pricePrompt, priceCompletion, contextWindow));
            }
        }
        if (list.isEmpty()) {
            // Legacy single-provider configuration
            list.add(new ModelConfig("default", legacyModel(), legacyApiKey(), legacyBaseUrl(),
                    null, null, null));
        }
        return list;
    }

    /** Optional USD-per-1M-tokens price; blank or invalid → null. */
    private Double getPrice(String key) {
        String raw = get(key, null);
        if (raw == null) {
            return null;
        }
        try {
            return Double.parseDouble(raw);
        } catch (NumberFormatException e) {
            return null;
        }
    }

    /** Optional integer property; blank or invalid → null. */
    private Integer getIntOrNull(String key) {
        String raw = get(key, null);
        if (raw == null) {
            return null;
        }
        try {
            return Integer.parseInt(raw);
        } catch (NumberFormatException e) {
            return null;
        }
    }

    /** Default model group ({@code llm.default-model}); falls back to the first group. */
    public ModelConfig defaultModelConfig() {
        return resolve(get("llm.default-model", null), modelConfigs());
    }

    /** Model group for batch-result analysis ({@code llm.analysis.model});
     * falls back to the default group. */
    public ModelConfig analysisModelConfig() {
        List<ModelConfig> list = modelConfigs();
        String ref = get("llm.analysis.model", null);
        if (ref != null) {
            return resolve(ref, list);
        }
        return resolve(get("llm.default-model", null), list);
    }

    /** Resolves a group reference — by group id or by model name. */
    private static ModelConfig resolve(String ref, List<ModelConfig> list) {
        if (ref != null) {
            for (ModelConfig c : list) {
                if (c.id().equalsIgnoreCase(ref) || c.model().equalsIgnoreCase(ref)) {
                    return c;
                }
            }
        }
        return list.get(0);
    }

    // ----- Legacy single-provider keys (fallback when no groups configured) -----

    private boolean isLocalServer() {
        String p = get("llm.provider", "cloud");
        return "local-server".equalsIgnoreCase(p) || "local".equalsIgnoreCase(p);
    }

    private String legacyApiKey() {
        if (isLocalServer()) {
            return get("llm.local.key", "not-required");
        }
        return require("llm.api.key");
    }

    private String legacyBaseUrl() {
        if (isLocalServer()) {
            return get("llm.local.base-url", "http://localhost:1234/v1");
        }
        return get("llm.api.base-url", "https://openrouter.ai/api/v1");
    }

    private String legacyModel() {
        if (isLocalServer()) {
            return require("llm.local.model");
        }
        return get("llm.model", "openai/gpt-4o-mini");
    }

    public int serverPort() {
        return getInt("server.port", 8080);
    }
}
