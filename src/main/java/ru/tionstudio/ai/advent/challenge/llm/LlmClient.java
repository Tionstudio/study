package ru.tionstudio.ai.advent.challenge.llm;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import ru.tionstudio.ai.advent.challenge.config.AppConfig;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

/**
 * Client for an OpenAI-compatible chat completions API
 * (OpenAI, OpenRouter, and any other provider exposing
 * {@code POST {baseUrl}/chat/completions}).
 */
public class LlmClient {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final HttpClient httpClient;
    private final String apiKey;
    private final String endpoint;
    private final String model;

    public LlmClient(AppConfig config) {
        this.apiKey = config.apiKey();
        this.endpoint = config.baseUrl().replaceAll("/+$", "") + "/chat/completions";
        this.model = config.model();
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(15))
                .build();
    }

    /**
     * Sends a single user message to the LLM and returns the assistant's reply text.
     *
     * @throws IOException          on network errors or non-2xx API responses
     * @throws InterruptedException if the request is interrupted
     */
    public String chat(String userMessage) throws IOException, InterruptedException {
        ObjectNode payload = MAPPER.createObjectNode();
        payload.put("model", model);
        ArrayNode messages = payload.putArray("messages");
        messages.addObject().put("role", "user").put("content", userMessage);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(endpoint))
                .timeout(Duration.ofSeconds(120))
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer " + apiKey)
                .POST(HttpRequest.BodyPublishers.ofString(MAPPER.writeValueAsString(payload)))
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() / 100 != 2) {
            throw new IOException("LLM API returned HTTP " + response.statusCode() + ": " + response.body());
        }

        JsonNode root = MAPPER.readTree(response.body());
        JsonNode content = root.path("choices").path(0).path("message").path("content");
        if (content.isMissingNode() || !content.isTextual()) {
            throw new IOException("Unexpected LLM API response format: " + response.body());
        }
        return content.asText();
    }
}
