package ru.tionstudio.ai.advent.challenge.server;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import ru.tionstudio.ai.advent.challenge.llm.LlmClient;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.Executors;

/**
 * Embedded HTTP server:
 * <ul>
 *   <li>{@code POST /api/chat} — accepts {@code {"message": "..."}} JSON,
 *       forwards it to the LLM and returns {@code {"reply": "..."}};</li>
 *   <li>everything else — serves static files from the {@code web/} directory
 *       (the simple JS chat page).</li>
 * </ul>
 */
public class ChatServer {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final Path WEB_ROOT = Path.of("web");

    private static final Map<String, String> CONTENT_TYPES = Map.of(
            ".html", "text/html; charset=utf-8",
            ".css", "text/css; charset=utf-8",
            ".js", "text/javascript; charset=utf-8",
            ".json", "application/json; charset=utf-8",
            ".png", "image/png",
            ".svg", "image/svg+xml",
            ".ico", "image/x-icon"
    );

    private final int port;
    private final LlmClient llmClient;

    public ChatServer(int port, LlmClient llmClient) {
        this.port = port;
        this.llmClient = llmClient;
    }

    public void start() throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);
        server.createContext("/api/chat", this::handleChat);
        server.createContext("/", this::handleStatic);
        server.setExecutor(Executors.newVirtualThreadPerTaskExecutor());
        server.start();
        System.out.println("Server started: http://localhost:" + port + "/");
    }

    private void handleChat(HttpExchange exchange) throws IOException {
        try {
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                sendJson(exchange, 405, errorJson("Method not allowed. Use POST."));
                return;
            }

            String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            JsonNode json = MAPPER.readTree(body);
            String message = json.path("message").asText("").trim();
            if (message.isEmpty()) {
                sendJson(exchange, 400, errorJson("Field 'message' must not be empty."));
                return;
            }

            String reply = llmClient.chat(message);

            ObjectNode responseJson = MAPPER.createObjectNode();
            responseJson.put("reply", reply);
            sendJson(exchange, 200, MAPPER.writeValueAsString(responseJson));
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            sendJson(exchange, 500, errorJson("Request was interrupted."));
        } catch (Exception e) {
            sendJson(exchange, 500, errorJson("LLM request failed: " + e.getMessage()));
        }
    }

    private void handleStatic(HttpExchange exchange) throws IOException {
        String path = exchange.getRequestURI().getPath();
        if (path.equals("/")) {
            path = "/index.html";
        }

        // Guard against path traversal (../../etc)
        Path file = WEB_ROOT.resolve(path.substring(1)).normalize();
        if (!file.startsWith(WEB_ROOT.normalize()) || !Files.isRegularFile(file)) {
            sendText(exchange, 404, "Not found: " + path);
            return;
        }

        String name = file.getFileName().toString();
        String contentType = "application/octet-stream";
        int dot = name.lastIndexOf('.');
        if (dot >= 0) {
            contentType = CONTENT_TYPES.getOrDefault(name.substring(dot), contentType);
        }

        byte[] bytes;
        try (InputStream in = Files.newInputStream(file)) {
            bytes = in.readAllBytes();
        }
        exchange.getResponseHeaders().set("Content-Type", contentType);
        exchange.sendResponseHeaders(200, bytes.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }

    private static String errorJson(String message) {
        ObjectNode node = MAPPER.createObjectNode();
        node.put("error", message);
        return node.toString();
    }

    private static void sendJson(HttpExchange exchange, int status, String json) throws IOException {
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        exchange.sendResponseHeaders(status, bytes.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }

    private static void sendText(HttpExchange exchange, int status, String text) throws IOException {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "text/plain; charset=utf-8");
        exchange.sendResponseHeaders(status, bytes.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }
}
