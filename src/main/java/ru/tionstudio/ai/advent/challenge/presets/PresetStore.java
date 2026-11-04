package ru.tionstudio.ai.advent.challenge.presets;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.List;
import java.util.logging.Logger;

/**
 * Local SQLite-backed storage for named settings presets
 * (the web UI "Сохранить" / preset-selector feature).
 *
 * <p>On first start the store seeds itself with the default example
 * presets. User presets are upserted by name; the {@code is_default}
 * flag of a record is preserved on overwrite.</p>
 */
public class PresetStore {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final Logger LOG = Logger.getLogger(PresetStore.class.getName());

    /** Maximum number of user-saved presets (default example presets are not counted). */
    public static final int MAX_USER_PRESETS = 100;

    /** Thrown when saving a NEW preset would exceed {@link #MAX_USER_PRESETS}. */
    public static class PresetLimitExceededException extends Exception {
        public PresetLimitExceededException(int max) {
            super("Достигнут лимит: не более " + max + " пользовательских наборов. "
                    + "Перезапишите существующий — сохраните с тем же названием.");
        }
    }

    private record DefaultPreset(String name, String settingsJson) {
    }

    /** The example presets from the settings guide, seeded on first start. */
    private static final List<DefaultPreset> DEFAULTS = List.of(
            new DefaultPreset("Эталон (всё по умолчанию)", "{}"),
            new DefaultPreset("Markdown-таблица пород",
                    "{\"format\":\"markdown-table\"}"),
            new DefaultPreset("JSON + JSONPath (названия пород)",
                    "{\"format\":\"json\",\"formatDescription\":\"{\\\"breeds\\\": [{\\\"name\\\": string, \\\"origin\\\": string, \\\"coat\\\": string}]}\",\"parseMode\":\"jsonpath\",\"parseExpression\":\"$.breeds[*].name\"}"),
            new DefaultPreset("XML + XPath (первая порода)",
                    "{\"format\":\"xml\",\"formatDescription\":\"<breeds><breed><name>...</name><description>...</description></breed></breeds>\",\"parseMode\":\"xpath\",\"parseExpression\":\"/breeds/breed[1]/name/text()\"}"),
            new DefaultPreset("YAML + лимит 1500 (размышления low)",
                    "{\"format\":\"yaml\",\"maxTokens\":1500,\"reasoningEffort\":\"low\"}"),
            new DefaultPreset("Chain-of-Thought (размышления high)",
                    "{\"format\":\"chain-of-thought\",\"reasoningEffort\":\"high\"}"),
            new DefaultPreset("Детская энциклопедия + стоп «Конец»",
                    "{\"format\":\"styled\",\"formatDescription\":\"детская энциклопедия для семилетнего читателя\",\"stopSequences\":\"Конец\"}")
    );

    private final String jdbcUrl;

    public PresetStore(Path dbPath) {
        this.jdbcUrl = "jdbc:sqlite:" + dbPath.toAbsolutePath();
        init();
        LOG.info("Preset store ready: " + dbPath.toAbsolutePath());
    }

    private Connection connect() throws SQLException {
        return DriverManager.getConnection(jdbcUrl);
    }

    private void init() {
        try (Connection c = connect(); Statement st = c.createStatement()) {
            st.executeUpdate("""
                    CREATE TABLE IF NOT EXISTS presets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE,
                        settings TEXT NOT NULL,
                        is_default INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL DEFAULT (datetime('now'))
                    )
                    """);
            try (ResultSet rs = st.executeQuery("SELECT COUNT(*) FROM presets")) {
                if (rs.next() && rs.getInt(1) == 0) {
                    seed(c);
                }
            }
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to initialize presets database", e);
        }
    }

    private void seed(Connection c) throws SQLException {
        try (PreparedStatement ps = c.prepareStatement(
                "INSERT INTO presets (name, settings, is_default) VALUES (?, ?, 1)")) {
            for (DefaultPreset p : DEFAULTS) {
                ps.setString(1, p.name());
                ps.setString(2, p.settingsJson());
                ps.addBatch();
            }
            ps.executeBatch();
        }
        LOG.info("Seeded " + DEFAULTS.size() + " default presets");
    }

    /** @return JSON: {"presets": [{"id", "name", "isDefault", "settings"}...]} */
    public String listJson() throws SQLException {
        ObjectNode root = MAPPER.createObjectNode();
        ArrayNode arr = root.putArray("presets");
        try (Connection c = connect();
             PreparedStatement ps = c.prepareStatement(
                     "SELECT id, name, settings, is_default FROM presets"
                             + " ORDER BY is_default, name COLLATE NOCASE");
             ResultSet rs = ps.executeQuery()) {
            while (rs.next()) {
                arr.add(toJson(rs));
            }
        }
        return root.toString();
    }

    /**
     * Upserts a preset by name (the is_default flag is preserved on overwrite).
     * Saving a NEW name is limited to {@link #MAX_USER_PRESETS} user presets;
     * overwriting an existing name is always allowed.
     *
     * @return JSON of the saved record
     * @throws PresetLimitExceededException if a new preset would exceed the limit
     */
    public String saveJson(String name, JsonNode settings)
            throws SQLException, PresetLimitExceededException {
        try (Connection c = connect()) {
            boolean exists;
            try (PreparedStatement ps = c.prepareStatement("SELECT 1 FROM presets WHERE name = ?")) {
                ps.setString(1, name);
                try (ResultSet rs = ps.executeQuery()) {
                    exists = rs.next();
                }
            }
            if (!exists) {
                int userCount;
                try (PreparedStatement ps = c.prepareStatement(
                        "SELECT COUNT(*) FROM presets WHERE is_default = 0");
                     ResultSet rs = ps.executeQuery()) {
                    rs.next();
                    userCount = rs.getInt(1);
                }
                if (userCount >= MAX_USER_PRESETS) {
                    LOG.warning("Preset limit exceeded (" + MAX_USER_PRESETS + "), rejected new preset: '" + name + "'");
                    throw new PresetLimitExceededException(MAX_USER_PRESETS);
                }
            }
            try (PreparedStatement ps = c.prepareStatement("""
                    INSERT INTO presets (name, settings, is_default) VALUES (?, ?, 0)
                    ON CONFLICT(name) DO UPDATE SET settings = excluded.settings
                    """)) {
                ps.setString(1, name);
                ps.setString(2, MAPPER.writeValueAsString(settings));
                ps.executeUpdate();
            }
            try (PreparedStatement ps = c.prepareStatement(
                    "SELECT id, name, settings, is_default FROM presets WHERE name = ?")) {
                ps.setString(1, name);
                try (ResultSet rs = ps.executeQuery()) {
                    if (!rs.next()) {
                        throw new SQLException("Preset not found after save: " + name);
                    }
                    return toJson(rs).toString();
                }
            }
        } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
            throw new SQLException("Failed to serialize preset settings", e);
        }
    }

    /**
     * Deletes a preset by id (both user and default presets can be deleted).
     *
     * @return {@code true} if a record was deleted, {@code false} if not found
     */
    public boolean delete(long id) throws SQLException {
        try (Connection c = connect();
             PreparedStatement ps = c.prepareStatement("DELETE FROM presets WHERE id = ?")) {
            ps.setLong(1, id);
            int deleted = ps.executeUpdate();
            if (deleted > 0) {
                LOG.info("Preset deleted: id=" + id);
            }
            return deleted > 0;
        }
    }

    private static ObjectNode toJson(ResultSet rs) throws SQLException {
        ObjectNode p = MAPPER.createObjectNode();
        p.put("id", rs.getLong("id"));
        p.put("name", rs.getString("name"));
        p.put("isDefault", rs.getInt("is_default") == 1);
        try {
            p.set("settings", MAPPER.readTree(rs.getString("settings")));
        } catch (Exception e) {
            p.set("settings", MAPPER.createObjectNode());
        }
        return p;
    }
}
