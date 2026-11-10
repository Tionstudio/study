package ru.tionstudio.ai.advent.challenge.agent;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

/**
 * File/time/calculation tools available to the LLM agent.
 *
 * <p>All file operations are sandboxed to a single working directory
 * ({@code test/}): only plain file names are accepted — no path separators,
 * no "..", no absolute paths — and the resolved path is re-checked against
 * the sandbox root.</p>
 */
public class AgentTools {

    /** Outcome of one tool execution; {@code text} goes back to the LLM. */
    public record ToolResult(boolean ok, String text) {
    }

    /** Long files are truncated before being handed to the LLM. */
    private static final int MAX_READ_CHARS = 20_000;
    private static final DateTimeFormatter TIME_FMT =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
    /** System-time tool reports milliseconds too (used e.g. for dice rolls). */
    private static final DateTimeFormatter TIME_FMT_MS =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS");

    private final Path workDir;

    public AgentTools(Path workDir) throws IOException {
        this.workDir = workDir.toAbsolutePath().normalize();
        Files.createDirectories(this.workDir);
    }

    public Path workDir() {
        return workDir;
    }

    /** Whether the tool mutates data and therefore needs user permission. */
    public static boolean requiresPermission(String tool) {
        return "write_file".equals(tool) || "append_file".equals(tool)
                || "delete_file".equals(tool);
    }

    /** Dispatches a validated tool call. */
    public ToolResult execute(String tool, String name, String content, String expression) {
        return switch (tool) {
            case "get_system_time" -> getSystemTime();
            case "list_files" -> listFiles();
            case "read_file" -> readFile(name);
            case "write_file" -> writeFile(name, content);
            case "append_file" -> appendFile(name, content);
            case "delete_file" -> deleteFile(name);
            case "calculate" -> calculate(expression);
            default -> new ToolResult(false, "Неизвестный инструмент: " + tool);
        };
    }

    public ToolResult getSystemTime() {
        LocalDateTime now = LocalDateTime.now();
        return new ToolResult(true, "Текущее системное время сервера: "
                + TIME_FMT_MS.format(now) + " (" + ZoneId.systemDefault().getId() + ")");
    }

    public ToolResult listFiles() {
        try (Stream<Path> stream = Files.list(workDir)) {
            List<String> lines = new ArrayList<>();
            stream.filter(Files::isRegularFile)
                    .sorted()
                    .forEach(p -> {
                        try {
                            lines.add(p.getFileName()
                                    + " — " + Files.size(p) + " байт, изменён "
                                    + TIME_FMT.format(LocalDateTime.ofInstant(
                                            Files.getLastModifiedTime(p).toInstant(),
                                            ZoneId.systemDefault())));
                        } catch (IOException e) {
                            lines.add(p.getFileName() + " — (ошибка чтения атрибутов)");
                        }
                    });
            if (lines.isEmpty()) {
                return new ToolResult(true, "Рабочая директория пуста (файлов нет).");
            }
            return new ToolResult(true, "Файлы в рабочей директории (" + lines.size() + "):\n"
                    + String.join("\n", lines));
        } catch (IOException e) {
            return new ToolResult(false, "Не удалось получить список файлов: " + e.getMessage());
        }
    }

    public ToolResult readFile(String name) {
        Path file;
        try {
            file = resolve(name);
        } catch (IllegalArgumentException e) {
            return new ToolResult(false, e.getMessage());
        }
        if (!Files.isRegularFile(file)) {
            return new ToolResult(false, "Файл не найден: " + name);
        }
        try {
            String text = Files.readString(file, StandardCharsets.UTF_8);
            if (text.length() > MAX_READ_CHARS) {
                text = text.substring(0, MAX_READ_CHARS)
                        + "\n… (файл обрезан до " + MAX_READ_CHARS + " символов)";
            }
            return new ToolResult(true, "Содержимое файла " + name + ":\n" + text);
        } catch (IOException e) {
            return new ToolResult(false, "Не удалось прочитать файл " + name + ": " + e.getMessage());
        }
    }

    public ToolResult writeFile(String name, String content) {
        Path file;
        try {
            file = resolve(name);
        } catch (IllegalArgumentException e) {
            return new ToolResult(false, e.getMessage());
        }
        try {
            Files.writeString(file, content, StandardCharsets.UTF_8);
            return new ToolResult(true, "Файл " + name + " записан ("
                    + content.length() + " символов).");
        } catch (IOException e) {
            return new ToolResult(false, "Не удалось записать файл " + name + ": " + e.getMessage());
        }
    }

    public ToolResult appendFile(String name, String content) {
        Path file;
        try {
            file = resolve(name);
        } catch (IllegalArgumentException e) {
            return new ToolResult(false, e.getMessage());
        }
        try {
            Files.writeString(file, content, StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
            return new ToolResult(true, "В файл " + name + " дописано "
                    + content.length() + " символов.");
        } catch (IOException e) {
            return new ToolResult(false, "Не удалось дописать файл " + name + ": " + e.getMessage());
        }
    }

    public ToolResult deleteFile(String name) {
        Path file;
        try {
            file = resolve(name);
        } catch (IllegalArgumentException e) {
            return new ToolResult(false, e.getMessage());
        }
        if (!Files.isRegularFile(file)) {
            return new ToolResult(false, "Файл не найден: " + name);
        }
        try {
            Files.delete(file);
            return new ToolResult(true, "Файл " + name + " удалён.");
        } catch (IOException e) {
            return new ToolResult(false, "Не удалось удалить файл " + name + ": " + e.getMessage());
        }
    }

    /**
     * Safe arithmetic evaluator: numbers, + - * / % ^, parentheses,
     * unary plus/minus. No names, no calls — a recursive descent parser.
     */
    public ToolResult calculate(String expression) {
        try {
            double value = new ExprParser(expression).parse();
            String out;
            if (value == Math.rint(value) && Math.abs(value) < 1e15) {
                out = String.valueOf((long) value);
            } else {
                out = String.valueOf(value);
            }
            return new ToolResult(true, expression + " = " + out);
        } catch (IllegalArgumentException e) {
            return new ToolResult(false, "Не удалось вычислить выражение: " + e.getMessage());
        }
    }

    /**
     * Resolves a plain file name inside the sandbox. Rejects path separators,
     * "..", drive letters and anything that would escape the working directory.
     */
    private Path resolve(String name) {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("Имя файла не должно быть пустым.");
        }
        String trimmed = name.trim();
        if (trimmed.length() > 255) {
            throw new IllegalArgumentException("Имя файла слишком длинное (максимум 255 символов).");
        }
        if (trimmed.contains("..") || trimmed.contains("/") || trimmed.contains("\\")
                || trimmed.contains(":")) {
            throw new IllegalArgumentException("Разрешено только простое имя файла "
                    + "без путей и '..': " + trimmed);
        }
        Path file = workDir.resolve(trimmed).normalize();
        if (!file.startsWith(workDir)) {
            throw new IllegalArgumentException("Имя файла выходит за пределы рабочей директории.");
        }
        return file;
    }

    /** Recursive descent parser for {@link #calculate}. */
    private static final class ExprParser {
        private final String s;
        private int pos;

        ExprParser(String s) {
            this.s = s == null ? "" : s;
        }

        double parse() {
            double v = expr();
            skipWs();
            if (pos != s.length()) {
                throw new IllegalArgumentException("лишние символы на позиции " + pos
                        + ": '" + s.substring(pos) + "'");
            }
            return v;
        }

        private void skipWs() {
            while (pos < s.length() && Character.isWhitespace(s.charAt(pos))) {
                pos++;
            }
        }

        private boolean eat(char c) {
            skipWs();
            if (pos < s.length() && s.charAt(pos) == c) {
                pos++;
                return true;
            }
            return false;
        }

        private double expr() { // + -
            double v = term();
            while (true) {
                if (eat('+')) {
                    v += term();
                } else if (eat('-')) {
                    v -= term();
                } else {
                    return v;
                }
            }
        }

        private double term() { // * / %
            double v = factor();
            while (true) {
                if (eat('*')) {
                    v *= factor();
                } else if (eat('/')) {
                    v /= factor();
                } else if (eat('%')) {
                    v %= factor();
                } else {
                    return v;
                }
            }
        }

        private double factor() { // unary + -, power (right-assoc)
            skipWs();
            if (eat('+')) {
                return factor();
            }
            if (eat('-')) {
                return -factor();
            }
            double base = primary();
            if (eat('^')) {
                return Math.pow(base, factor());
            }
            return base;
        }

        private double primary() {
            skipWs();
            if (eat('(')) {
                double v = expr();
                if (!eat(')')) {
                    throw new IllegalArgumentException("не закрыта скобка на позиции " + pos);
                }
                return v;
            }
            int start = pos;
            while (pos < s.length()
                    && (Character.isDigit(s.charAt(pos)) || s.charAt(pos) == '.')) {
                pos++;
            }
            if (start == pos) {
                throw new IllegalArgumentException("ожидалось число на позиции " + pos);
            }
            try {
                return Double.parseDouble(s.substring(start, pos));
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("некорректное число: '"
                        + s.substring(start, pos) + "'");
            }
        }
    }
}
