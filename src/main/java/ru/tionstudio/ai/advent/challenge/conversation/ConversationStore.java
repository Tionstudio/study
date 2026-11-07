package ru.tionstudio.ai.advent.challenge.conversation;

import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.logging.Logger;

/**
 * SQLite-backed storage of chat conversations for the context mode
 * ("Сохранять контекст" in the web UI). Lives in the same database file
 * as the settings presets ({@code settings.db}).
 *
 * <p>A conversation holds the full message history plus an optional
 * LLM-generated {@code summary} that compresses the first
 * {@code summaryUpto} messages (see {@code ConversationManager}).</p>
 */
public class ConversationStore {

    private static final Logger LOG = Logger.getLogger(ConversationStore.class.getName());

    public record StoredMessage(String role, String content) {
    }

    /**
     * A conversation snapshot.
     *
     * @param summary     compressed text of the oldest messages (may be empty)
     * @param summaryUpto how many of the oldest messages the summary covers
     * @param messages    full history in chronological order
     */
    public record Conversation(String id, String summary, int summaryUpto,
                               List<StoredMessage> messages) {
    }

    private final String jdbcUrl;

    public ConversationStore(Path dbPath) {
        this.jdbcUrl = "jdbc:sqlite:" + dbPath.toAbsolutePath();
        init();
        LOG.info("Conversation store ready: " + dbPath.toAbsolutePath());
    }

    private Connection connect() throws SQLException {
        return DriverManager.getConnection(jdbcUrl);
    }

    private void init() {
        try (Connection c = connect(); Statement st = c.createStatement()) {
            st.executeUpdate("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id TEXT PRIMARY KEY,
                        summary TEXT NOT NULL DEFAULT '',
                        summary_upto INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL DEFAULT (datetime('now')),
                        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                    )
                    """);
            st.executeUpdate("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT (datetime('now'))
                    )
                    """);
            st.executeUpdate(
                    "CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id)");
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to initialize conversations database", e);
        }
    }

    /** Creates an empty conversation and returns its generated id. */
    public String create() {
        String id = UUID.randomUUID().toString();
        try (Connection c = connect();
             PreparedStatement ps = c.prepareStatement("INSERT INTO conversations (id) VALUES (?)")) {
            ps.setString(1, id);
            ps.executeUpdate();
            LOG.info("Conversation created: " + id);
            return id;
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to create conversation", e);
        }
    }

    public boolean exists(String id) {
        try (Connection c = connect();
             PreparedStatement ps = c.prepareStatement("SELECT 1 FROM conversations WHERE id = ?")) {
            ps.setString(1, id);
            try (ResultSet rs = ps.executeQuery()) {
                return rs.next();
            }
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to check conversation " + id, e);
        }
    }

    /** Appends a message (role: user | assistant) to the conversation history. */
    public void append(String conversationId, String role, String content) {
        try (Connection c = connect()) {
            try (PreparedStatement ps = c.prepareStatement(
                    "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)")) {
                ps.setString(1, conversationId);
                ps.setString(2, role);
                ps.setString(3, content);
                ps.executeUpdate();
            }
            try (PreparedStatement ps = c.prepareStatement(
                    "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?")) {
                ps.setString(1, conversationId);
                ps.executeUpdate();
            }
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to append message to " + conversationId, e);
        }
    }

    /** Loads the full conversation; returns {@code null} if the id is unknown. */
    public Conversation load(String id) {
        try (Connection c = connect()) {
            String summary = "";
            int summaryUpto = 0;
            try (PreparedStatement ps = c.prepareStatement(
                    "SELECT summary, summary_upto FROM conversations WHERE id = ?")) {
                ps.setString(1, id);
                try (ResultSet rs = ps.executeQuery()) {
                    if (!rs.next()) {
                        return null;
                    }
                    summary = rs.getString("summary");
                    summaryUpto = rs.getInt("summary_upto");
                }
            }
            List<StoredMessage> messages = new ArrayList<>();
            try (PreparedStatement ps = c.prepareStatement(
                    "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id")) {
                ps.setString(1, id);
                try (ResultSet rs = ps.executeQuery()) {
                    while (rs.next()) {
                        messages.add(new StoredMessage(rs.getString("role"), rs.getString("content")));
                    }
                }
            }
            return new Conversation(id, summary, summaryUpto, messages);
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to load conversation " + id, e);
        }
    }

    /** Replaces the summary and moves its coverage marker forward. */
    public void updateSummary(String id, String summary, int summaryUpto) {
        try (Connection c = connect();
             PreparedStatement ps = c.prepareStatement(
                     "UPDATE conversations SET summary = ?, summary_upto = ? WHERE id = ?")) {
            ps.setString(1, summary);
            ps.setInt(2, summaryUpto);
            ps.setString(3, id);
            ps.executeUpdate();
            LOG.info("Conversation " + id + ": summary updated, covers first "
                    + summaryUpto + " messages (" + summary.length() + " chars)");
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to update summary of " + id, e);
        }
    }

    /** Deletes the conversation and all its messages; returns true if it existed. */
    public boolean delete(String id) {
        try (Connection c = connect()) {
            try (PreparedStatement ps = c.prepareStatement(
                    "DELETE FROM messages WHERE conversation_id = ?")) {
                ps.setString(1, id);
                ps.executeUpdate();
            }
            try (PreparedStatement ps = c.prepareStatement(
                    "DELETE FROM conversations WHERE id = ?")) {
                ps.setString(1, id);
                boolean deleted = ps.executeUpdate() > 0;
                if (deleted) {
                    LOG.info("Conversation deleted: " + id);
                }
                return deleted;
            }
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to delete conversation " + id, e);
        }
    }
}
