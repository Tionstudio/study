package ru.tionstudio.ai.advent.challenge;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;
import java.util.logging.Handler;
import java.util.logging.LogRecord;
import java.util.logging.SimpleFormatter;

/**
 * Log handler that keeps only the last {@code maxRecords} log records in
 * a single file. The file is rewritten on every record (a hundred short
 * lines is trivially cheap), so it never grows beyond the ring size and
 * always shows the most recent activity.
 */
public class RingFileLogHandler extends Handler {

    private final Path file;
    private final int maxRecords;
    private final Deque<String> records = new ArrayDeque<>();
    private final SimpleFormatter formatter = new SimpleFormatter();

    public RingFileLogHandler(Path file, int maxRecords) {
        this.file = file;
        this.maxRecords = Math.max(1, maxRecords);
    }

    @Override
    public synchronized void publish(LogRecord record) {
        if (record == null || !isLoggable(record)) {
            return;
        }
        // formatter.format() ends with a newline; Files.write adds another —
        // strip ours to avoid blank lines between records
        records.addLast(formatter.format(record).replaceAll("\\r?\\n$", ""));
        while (records.size() > maxRecords) {
            records.removeFirst();
        }
        try {
            Files.write(file, List.copyOf(records), StandardCharsets.UTF_8);
        } catch (IOException e) {
            // Logging must never break the application
        }
    }

    @Override
    public void flush() {
        // everything is written eagerly in publish()
    }

    @Override
    public void close() {
        // nothing to release
    }
}
