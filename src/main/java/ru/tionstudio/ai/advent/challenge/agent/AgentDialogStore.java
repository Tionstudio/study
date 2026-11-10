package ru.tionstudio.ai.advent.challenge.agent;

import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

/**
 * SQLite-backed storage of agent-page dialogs. Lives in the same database
 * file as the settings presets ({@code settings.db}).
 *
 * <p>Schema is multi-dialog ready: every dialog is a row in
 * {@code agent_dialogs} and its transcript/history entries live in
 * {@code agent_dialog_entries}. The {@code agent_state} table holds the
 * pointer to the <em>current</em> dialog — the one the page restores after
 * an application restart. Listing/switching dialogs can be added later
 * without a schema change.</p>
 *
 * <p>Entry kinds:</p>
 * <ul>
 *   <li>UI transcript: {@code user}, {@code assistant}, {@code prose},
 *       {@code tool}, {@code plan}, {@code permission}, {@code note} —
 *       everything the page needs to re-render the chat;</li>
 *   <li>{@code history} — LLM context entries ({role, content}); replaying
 *       them in order restores the conversation so the dialog can be
 *       continued.</li>
 * </ul>
 */
public class AgentDialogStore {

    private static final Logger LOG = Logger.getLogger(AgentDialogStore.class.getName());

    private static final String STATE_CURRENT_DIALOG = "current_dialog";

    /** One stored entry: kind plus its JSON payload. */
    public record StoredEntry(String kind, String payload) {
    }

    /** Dialog metadata (for a future multi-dialog list). */
    public record DialogInfo(String id, String title, String createdAt, String updatedAt) {
    }

    private final String jdbcUrl;

    public AgentDialogStore(Path dbPath) {
        this.jdbcUrl = "jdbc:sqlite:" + dbPath.toAbsolutePath();
        init();
        LOG.info("Agent dialog store ready: " + dbPath.toAbsolutePath());
    }

    private Connection connect() throws SQLException {
        return DriverManager.getConnection(jdbcUrl);
    }

    private void init() {
        try (Connection c = connect(); Statement st = c.createStatement()) {
            st.executeUpdate("""
                    CREATE TABLE IF NOT EXISTS agent_dialogs (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL DEFAULT (datetime('now')),
                        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                    )
                    """);
            st.executeUpdate("""
                    CREATE TABLE IF NOT EXISTS agent_dialog_entries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        dialog_id TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT (datetime('now'))
                    )
                    """);
            st.executeUpdate("CREATE INDEX IF NOT EXISTS idx_agent_entries_dialog "
                    + "ON agent_dialog_entries(dialog_id, id)");
            st.executeUpdate("""
                    CREATE TABLE IF NOT EXISTS agent_state (
                        key TEXT PRIMARY KEY,
                        value TEXT
                    )
                    """);
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to initialize agent dialogs database", e);
        }
    }

    // ----- dialogs -----

    /** Creates a dialog row with the given id and marks it as current. */
    public synchronized void createDialog(String id, String title) {
        try (Connection c = connect();
             PreparedStatement ps = c.prepareStatement(
                     "INSERT OR IGNORE INTO agent_dialogs (id, title) VALUES (?, ?)")) {
            ps.setString(1, id);
            ps.setString(2, title == null ? "" : title);
            ps.executeUpdate();
            setCurrentDialog(id);
            LOG.info("Agent dialog created: " + id + " («" + title + "»)");
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to create agent dialog " + id, e);
        }
    }

    public synchronized boolean dialogExists(String id) {
        try (Connection c = connect();
             PreparedStatement ps = c.prepareStatement(
                     "SELECT 1 FROM agent_dialogs WHERE id = ?")) {
            ps.setString(1, id);
            try (ResultSet rs = ps.executeQuery()) {
                return rs.next();
            }
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to check agent dialog " + id, e);
        }
    }

    /** All dialogs, most recently updated first (for a future dialog list). */
    public synchronized List<DialogInfo> listDialogs() {
        List<DialogInfo> result = new ArrayList<>();
        try (Connection c = connect();
             PreparedStatement ps = c.prepareStatement(
                     "SELECT id, title, created_at, updated_at FROM agent_dialogs "
                             + "ORDER BY updated_at DESC")) {
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    result.add(new DialogInfo(rs.getString("id"), rs.getString("title"),
                            rs.getString("created_at"), rs.getString("updated_at")));
                }
            }
            return result;
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to list agent dialogs", e);
        }
    }

    // ----- current-dialog pointer -----

    /** Id of the dialog the agent page should restore, or {@code null}. */
    public synchronized String currentDialogId() {
        try (Connection c = connect();
             PreparedStatement ps = c.prepareStatement(
                     "SELECT value FROM agent_state WHERE key = ?")) {
            ps.setString(1, STATE_CURRENT_DIALOG);
            try (ResultSet rs = ps.executeQuery()) {
                return rs.next() ? rs.getString("value") : null;
            }
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to read current agent dialog", e);
        }
    }

    public synchronized void setCurrentDialog(String id) {
        try (Connection c = connect();
             PreparedStatement ps = c.prepareStatement(
                     "INSERT INTO agent_state (key, value) VALUES (?, ?) "
                             + "ON CONFLICT(key) DO UPDATE SET value = excluded.value")) {
            ps.setString(1, STATE_CURRENT_DIALOG);
            ps.setString(2, id);
            ps.executeUpdate();
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to set current agent dialog", e);
        }
    }

    /** Clears the pointer ("Новый диалог") — the stored dialog itself is kept. */
    public synchronized void clearCurrentDialog() {
        try (Connection c = connect();
             PreparedStatement ps = c.prepareStatement(
                     "DELETE FROM agent_state WHERE key = ?")) {
            ps.setString(1, STATE_CURRENT_DIALOG);
            ps.executeUpdate();
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to clear current agent dialog", e);
        }
    }

    // ----- entries -----

    /** Appends a transcript/history entry to the dialog. */
    public synchronized void appendEntry(String dialogId, String kind, String payload) {
        try (Connection c = connect()) {
            try (PreparedStatement ps = c.prepareStatement(
                    "INSERT INTO agent_dialog_entries (dialog_id, kind, payload) "
                            + "VALUES (?, ?, ?)")) {
                ps.setString(1, dialogId);
                ps.setString(2, kind);
                ps.setString(3, payload);
                ps.executeUpdate();
            }
            try (PreparedStatement ps = c.prepareStatement(
                    "UPDATE agent_dialogs SET updated_at = datetime('now') WHERE id = ?")) {
                ps.setString(1, dialogId);
                ps.executeUpdate();
            }
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to append entry to agent dialog " + dialogId, e);
        }
    }

    /** All entries of one kind (e.g. only "history") in chronological order. */
    public synchronized List<StoredEntry> loadEntries(String dialogId, String kind) {
        List<StoredEntry> result = new ArrayList<>();
        try (Connection c = connect();
             PreparedStatement ps = c.prepareStatement(
                     "SELECT kind, payload FROM agent_dialog_entries "
                             + "WHERE dialog_id = ? AND kind = ? ORDER BY id")) {
            ps.setString(1, dialogId);
            ps.setString(2, kind);
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    result.add(new StoredEntry(rs.getString("kind"), rs.getString("payload")));
                }
            }
            return result;
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to load entries of agent dialog " + dialogId, e);
        }
    }

    /** All entries except the given kind (e.g. everything but "history"). */
    public synchronized List<StoredEntry> loadEntriesExcept(String dialogId, String excludedKind) {
        List<StoredEntry> result = new ArrayList<>();
        try (Connection c = connect();
             PreparedStatement ps = c.prepareStatement(
                     "SELECT kind, payload FROM agent_dialog_entries "
                             + "WHERE dialog_id = ? AND kind <> ? ORDER BY id")) {
            ps.setString(1, dialogId);
            ps.setString(2, excludedKind);
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    result.add(new StoredEntry(rs.getString("kind"), rs.getString("payload")));
                }
            }
            return result;
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to load entries of agent dialog " + dialogId, e);
        }
    }

    /** Deletes the dialog and all its entries; returns true if it existed. */
    public synchronized boolean deleteDialog(String id) {
        try (Connection c = connect()) {
            try (PreparedStatement ps = c.prepareStatement(
                    "DELETE FROM agent_dialog_entries WHERE dialog_id = ?")) {
                ps.setString(1, id);
                ps.executeUpdate();
            }
            try (PreparedStatement ps = c.prepareStatement(
                    "DELETE FROM agent_dialogs WHERE id = ?")) {
                ps.setString(1, id);
                boolean deleted = ps.executeUpdate() > 0;
                if (deleted) {
                    LOG.info("Agent dialog deleted: " + id);
                }
                return deleted;
            }
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to delete agent dialog " + id, e);
        }
    }
}
