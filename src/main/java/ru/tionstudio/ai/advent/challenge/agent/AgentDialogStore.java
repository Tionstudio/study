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

    /**
     * A saved checkpoint: a label plus the id of the last dialog entry at save
     * time (the fork boundary) and a JSON snapshot of mutable state that lives
     * outside the entries — the fact list (sticky mode) and the page settings.
     */
    public record Checkpoint(long id, String dialogId, String label, long entryId,
                             String stateJson, String createdAt) {
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
            st.executeUpdate("""
                    CREATE TABLE IF NOT EXISTS agent_facts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        dialog_id TEXT NOT NULL,
                        category TEXT NOT NULL,
                        key_norm TEXT NOT NULL,
                        key_display TEXT NOT NULL,
                        value TEXT NOT NULL,
                        updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                        UNIQUE(dialog_id, key_norm)
                    )
                    """);
            st.executeUpdate("""
                    CREATE TABLE IF NOT EXISTS agent_checkpoints (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        dialog_id TEXT NOT NULL,
                        label TEXT NOT NULL,
                        entry_id INTEGER NOT NULL,
                        state_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL DEFAULT (datetime('now'))
                    )
                    """);
            st.executeUpdate("CREATE INDEX IF NOT EXISTS idx_agent_checkpoints_dialog "
                    + "ON agent_checkpoints(dialog_id, id)");
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

    /** Deletes the dialog and all its entries, facts and checkpoints. */
    public synchronized boolean deleteDialog(String id) {
        try (Connection c = connect()) {
            try (PreparedStatement ps = c.prepareStatement(
                    "DELETE FROM agent_dialog_entries WHERE dialog_id = ?")) {
                ps.setString(1, id);
                ps.executeUpdate();
            }
            try (PreparedStatement ps = c.prepareStatement(
                    "DELETE FROM agent_facts WHERE dialog_id = ?")) {
                ps.setString(1, id);
                ps.executeUpdate();
            }
            try (PreparedStatement ps = c.prepareStatement(
                    "DELETE FROM agent_checkpoints WHERE dialog_id = ?")) {
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

    // ----- sticky facts (key-value memory, compression mode "sticky") -----

    /** Normalizes a fact key for deduplication: lower-case, ё→е, single spaces. */
    private static String normalizeFactKey(String key) {
        return key.toLowerCase().replace('ё', 'е').replaceAll("\\s+", " ").trim();
    }

    /** All facts of the dialog, oldest first. */
    public synchronized List<AgentProtocol.Fact> listFacts(String dialogId) {
        List<AgentProtocol.Fact> result = new ArrayList<>();
        try (Connection c = connect();
             PreparedStatement ps = c.prepareStatement(
                     "SELECT category, key_display, value FROM agent_facts "
                             + "WHERE dialog_id = ? ORDER BY updated_at, id")) {
            ps.setString(1, dialogId);
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    result.add(new AgentProtocol.Fact(
                            rs.getString(1), rs.getString(2), rs.getString(3)));
                }
            }
            return result;
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to load facts of dialog " + dialogId, e);
        }
    }

    /**
     * Upserts facts by normalized key and evicts the least recently updated
     * facts beyond {@code maxFacts}. Returns the full resulting list.
     */
    public synchronized List<AgentProtocol.Fact> upsertFacts(String dialogId,
            List<AgentProtocol.Fact> facts, int maxFacts) {
        try (Connection c = connect();
             PreparedStatement ps = c.prepareStatement("""
                     INSERT INTO agent_facts (dialog_id, category, key_norm, key_display, value)
                     VALUES (?, ?, ?, ?, ?)
                     ON CONFLICT(dialog_id, key_norm) DO UPDATE SET
                         category = excluded.category,
                         key_display = excluded.key_display,
                         value = excluded.value,
                         updated_at = datetime('now')
                     """)) {
            for (AgentProtocol.Fact f : facts) {
                ps.setString(1, dialogId);
                ps.setString(2, f.category());
                ps.setString(3, normalizeFactKey(f.key()));
                ps.setString(4, f.key());
                ps.setString(5, f.value());
                ps.addBatch();
            }
            ps.executeBatch();
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to upsert facts of dialog " + dialogId, e);
        }
        evictFacts(dialogId, maxFacts);
        return listFacts(dialogId);
    }
    /** Replaces the whole fact list (after LLM consolidation). */
    public synchronized List<AgentProtocol.Fact> replaceFacts(String dialogId,
            List<AgentProtocol.Fact> facts) {
        try (Connection c = connect()) {
            try (PreparedStatement ps = c.prepareStatement(
                    "DELETE FROM agent_facts WHERE dialog_id = ?")) {
                ps.setString(1, dialogId);
                ps.executeUpdate();
            }
            try (PreparedStatement ps = c.prepareStatement(
                    "INSERT INTO agent_facts (dialog_id, category, key_norm, key_display, value) "
                            + "VALUES (?, ?, ?, ?, ?)")) {
                for (AgentProtocol.Fact f : facts) {
                    ps.setString(1, dialogId);
                    ps.setString(2, f.category());
                    ps.setString(3, normalizeFactKey(f.key()));
                    ps.setString(4, f.key());
                    ps.setString(5, f.value());
                    ps.addBatch();
                }
                ps.executeBatch();
            }
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to replace facts of dialog " + dialogId, e);
        }
        return listFacts(dialogId);
    }

    /** Keeps only the {@code maxFacts} most recently updated facts of the dialog. */
    private void evictFacts(String dialogId, int maxFacts) {
        if (maxFacts <= 0) {
            return;
        }
        try (Connection c = connect();
             PreparedStatement ps = c.prepareStatement("""
                     DELETE FROM agent_facts WHERE dialog_id = ? AND id NOT IN (
                         SELECT id FROM agent_facts WHERE dialog_id = ?
                         ORDER BY updated_at DESC, id DESC LIMIT ?)
                     """)) {
            ps.setString(1, dialogId);
            ps.setString(2, dialogId);
            ps.setInt(3, maxFacts);
            ps.executeUpdate();
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to evict facts of dialog " + dialogId, e);
        }
    }

    // ----- checkpoints (fork a dialog from a saved snapshot) -----

    /** Creates a dialog row WITHOUT moving the current-dialog pointer (fork target). */
    public synchronized void createDialogQuiet(String id, String title) {
        try (Connection c = connect();
             PreparedStatement ps = c.prepareStatement(
                     "INSERT OR IGNORE INTO agent_dialogs (id, title) VALUES (?, ?)")) {
            ps.setString(1, id);
            ps.setString(2, title == null ? "" : title);
            ps.executeUpdate();
            LOG.info("Agent dialog created (fork): " + id + " («" + title + "»)");
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to create agent dialog " + id, e);
        }
    }

    /** Title of the dialog, or an empty string when unknown. */
    public synchronized String dialogTitle(String id) {
        try (Connection c = connect();
             PreparedStatement ps = c.prepareStatement(
                     "SELECT title FROM agent_dialogs WHERE id = ?")) {
            ps.setString(1, id);
            try (ResultSet rs = ps.executeQuery()) {
                return rs.next() ? rs.getString("title") : "";
            }
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to read title of dialog " + id, e);
        }
    }

    /** Id of the newest entry of the dialog — the fork boundary of a checkpoint. */
    public synchronized long maxEntryId(String dialogId) {
        try (Connection c = connect();
             PreparedStatement ps = c.prepareStatement(
                     "SELECT COALESCE(MAX(id), 0) FROM agent_dialog_entries WHERE dialog_id = ?")) {
            ps.setString(1, dialogId);
            try (ResultSet rs = ps.executeQuery()) {
                return rs.next() ? rs.getLong(1) : 0;
            }
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to read max entry id of dialog " + dialogId, e);
        }
    }

    /** Saves a checkpoint and returns it (with the generated id). */
    public synchronized Checkpoint saveCheckpoint(String dialogId, String label,
                                                  long entryId, String stateJson) {
        try (Connection c = connect();
             PreparedStatement ps = c.prepareStatement(
                     "INSERT INTO agent_checkpoints (dialog_id, label, entry_id, state_json) "
                             + "VALUES (?, ?, ?, ?)")) {
            ps.setString(1, dialogId);
            ps.setString(2, label);
            ps.setLong(3, entryId);
            ps.setString(4, stateJson == null ? "{}" : stateJson);
            ps.executeUpdate();
            try (Statement st = c.createStatement();
                 ResultSet rs = st.executeQuery("SELECT last_insert_rowid()")) {
                rs.next();
                long id = rs.getLong(1);
                LOG.info("Checkpoint " + id + " saved for dialog " + dialogId
                        + " («" + label + "», entry " + entryId + ")");
                return new Checkpoint(id, dialogId, label, entryId, stateJson, "");
            }
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to save checkpoint of dialog " + dialogId, e);
        }
    }

    /** All checkpoints of the dialog, oldest first. */
    public synchronized List<Checkpoint> listCheckpoints(String dialogId) {
        List<Checkpoint> result = new ArrayList<>();
        try (Connection c = connect();
             PreparedStatement ps = c.prepareStatement(
                     "SELECT id, dialog_id, label, entry_id, state_json, created_at "
                             + "FROM agent_checkpoints WHERE dialog_id = ? ORDER BY id")) {
            ps.setString(1, dialogId);
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    result.add(new Checkpoint(rs.getLong("id"), rs.getString("dialog_id"),
                            rs.getString("label"), rs.getLong("entry_id"),
                            rs.getString("state_json"), rs.getString("created_at")));
                }
            }
            return result;
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to list checkpoints of dialog " + dialogId, e);
        }
    }

    /** One checkpoint by id, or {@code null} when unknown. */
    public synchronized Checkpoint getCheckpoint(long id) {
        try (Connection c = connect();
             PreparedStatement ps = c.prepareStatement(
                     "SELECT id, dialog_id, label, entry_id, state_json, created_at "
                             + "FROM agent_checkpoints WHERE id = ?")) {
            ps.setLong(1, id);
            try (ResultSet rs = ps.executeQuery()) {
                return rs.next()
                        ? new Checkpoint(rs.getLong("id"), rs.getString("dialog_id"),
                                rs.getString("label"), rs.getLong("entry_id"),
                                rs.getString("state_json"), rs.getString("created_at"))
                        : null;
            }
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to read checkpoint " + id, e);
        }
    }

    /** Deletes a checkpoint; the dialog itself is untouched. */
    public synchronized boolean deleteCheckpoint(long id) {
        try (Connection c = connect();
             PreparedStatement ps = c.prepareStatement(
                     "DELETE FROM agent_checkpoints WHERE id = ?")) {
            ps.setLong(1, id);
            boolean deleted = ps.executeUpdate() > 0;
            if (deleted) {
                LOG.info("Checkpoint deleted: " + id);
            }
            return deleted;
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to delete checkpoint " + id, e);
        }
    }

    /**
     * Entry id of the fork boundary right AFTER turn {@code turn} (1-based,
     * turns are counted by transcript entries of kind {@code user}): the
     * newest entry of this dialog that precedes the first entry of turn
     * {@code turn + 1} — or simply the newest entry when {@code turn} is the
     * last one. Returns {@code -1} when the dialog has fewer turns.
     */
    public synchronized long entryIdAfterTurn(String dialogId, int turn) {
        try (Connection c = connect()) {
            List<Long> userIds = new ArrayList<>();
            try (PreparedStatement ps = c.prepareStatement(
                    "SELECT id FROM agent_dialog_entries "
                            + "WHERE dialog_id = ? AND kind = 'user' ORDER BY id")) {
                ps.setString(1, dialogId);
                try (ResultSet rs = ps.executeQuery()) {
                    while (rs.next()) {
                        userIds.add(rs.getLong(1));
                    }
                }
            }
            if (turn < 1 || turn > userIds.size()) {
                return -1;
            }
            if (turn == userIds.size()) {
                try (PreparedStatement ps = c.prepareStatement(
                        "SELECT COALESCE(MAX(id), 0) FROM agent_dialog_entries "
                                + "WHERE dialog_id = ?")) {
                    ps.setString(1, dialogId);
                    try (ResultSet rs = ps.executeQuery()) {
                        return rs.next() ? rs.getLong(1) : 0;
                    }
                }
            }
            long nextTurnStart = userIds.get(turn); // 0-based: first entry of turn+1
            try (PreparedStatement ps = c.prepareStatement(
                    "SELECT COALESCE(MAX(id), 0) FROM agent_dialog_entries "
                            + "WHERE dialog_id = ? AND id < ?")) {
                ps.setString(1, dialogId);
                ps.setLong(2, nextTurnStart);
                try (ResultSet rs = ps.executeQuery()) {
                    return rs.next() ? rs.getLong(1) : 0;
                }
            }
        } catch (SQLException e) {
            throw new IllegalStateException(
                    "Failed to compute turn boundary of dialog " + dialogId, e);
        }
    }

    /** Entries of one kind with id &le; {@code maxEntryId}, chronological. */
    public synchronized List<StoredEntry> loadEntriesUpTo(String dialogId, String kind,
                                                          long maxEntryId) {
        List<StoredEntry> result = new ArrayList<>();
        try (Connection c = connect();
             PreparedStatement ps = c.prepareStatement(
                     "SELECT kind, payload FROM agent_dialog_entries "
                             + "WHERE dialog_id = ? AND kind = ? AND id <= ? ORDER BY id")) {
            ps.setString(1, dialogId);
            ps.setString(2, kind);
            ps.setLong(3, maxEntryId);
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    result.add(new StoredEntry(rs.getString("kind"), rs.getString("payload")));
                }
            }
            return result;
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to load entries of dialog " + dialogId, e);
        }
    }

    /**
     * Copies all entries of the source dialog up to (and including)
     * {@code maxEntryId} into the target dialog — the materialized fork of a
     * checkpoint. New ids are assigned in chronological order.
     */
    public synchronized void copyEntriesUpTo(String srcDialogId, String dstDialogId,
                                             long maxEntryId) {
        try (Connection c = connect();
             PreparedStatement ps = c.prepareStatement(
                     "INSERT INTO agent_dialog_entries (dialog_id, kind, payload, created_at) "
                             + "SELECT ?, kind, payload, created_at FROM agent_dialog_entries "
                             + "WHERE dialog_id = ? AND id <= ? ORDER BY id")) {
            ps.setString(1, dstDialogId);
            ps.setString(2, srcDialogId);
            ps.setLong(3, maxEntryId);
            int copied = ps.executeUpdate();
            LOG.info("Forked dialog " + dstDialogId + ": " + copied
                    + " entries copied from " + srcDialogId + " (up to entry " + maxEntryId + ")");
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to copy entries of dialog " + srcDialogId, e);
        }
    }
}
