package ru.tionstudio.ai.advent.challenge.agent.v2;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import ru.tionstudio.ai.advent.challenge.agent.AgentProtocol;

import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.logging.Logger;

/** SQLite storage for the v2 agent: dialog ST, task WT and global agent config LT. */
public class MemoryStore {
    private static final Logger LOG = Logger.getLogger(MemoryStore.class.getName());
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final String CURRENT_DIALOG = "current_dialog";

    public record StoredEntry(String kind, String payload) { }
    public record DialogInfo(String id, String title, String createdAt, String updatedAt) { }

    private final String jdbcUrl;

    public MemoryStore(Path dbPath) {
        this.jdbcUrl = "jdbc:sqlite:" + dbPath.toAbsolutePath();
        init();
        LOG.info("Agent v2 memory store ready: " + dbPath.toAbsolutePath());
    }

    private Connection connect() throws SQLException {
        return DriverManager.getConnection(jdbcUrl);
    }

    private void init() {
        try (Connection c = connect(); Statement st = c.createStatement()) {
            st.executeUpdate("""
                    CREATE TABLE IF NOT EXISTS v2_dialogs (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL DEFAULT 'Диалог',
                        created_at TEXT NOT NULL DEFAULT (datetime('now')),
                        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                    )
                    """);
            st.executeUpdate("""
                    CREATE TABLE IF NOT EXISTS v2_entries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        dialog_id TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT (datetime('now'))
                    )
                    """);
            st.executeUpdate("CREATE INDEX IF NOT EXISTS idx_v2_entries_dialog "
                    + "ON v2_entries(dialog_id, id)");
            st.executeUpdate("""
                    CREATE TABLE IF NOT EXISTS v2_facts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        dialog_id TEXT NOT NULL,
                        category TEXT NOT NULL DEFAULT 'fact',
                        key_norm TEXT NOT NULL,
                        key_display TEXT NOT NULL,
                        value TEXT NOT NULL,
                        updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                        UNIQUE(dialog_id, key_norm)
                    )
                    """);
            st.executeUpdate("""
                    CREATE TABLE IF NOT EXISTS v2_working (
                        dialog_id TEXT NOT NULL,
                        field TEXT NOT NULL,
                        value TEXT NOT NULL DEFAULT '',
                        updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                        PRIMARY KEY(dialog_id, field)
                    )
                    """);
            st.executeUpdate("""
                    CREATE TABLE IF NOT EXISTS v2_agent_config (
                        id INTEGER PRIMARY KEY CHECK(id = 1),
                        content TEXT NOT NULL DEFAULT '{"rules":[]}',
                        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                    )
                    """);
            st.executeUpdate("""
                    CREATE TABLE IF NOT EXISTS v2_rule_candidates (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        candidate_key TEXT NOT NULL UNIQUE,
                        rule_text TEXT NOT NULL,
                        source_fact TEXT NOT NULL DEFAULT '',
                        source_dialog TEXT NOT NULL DEFAULT '',
                        operation TEXT NOT NULL DEFAULT 'add',
                        target_rule_id TEXT NOT NULL DEFAULT '',
                        base_text TEXT NOT NULL DEFAULT '',
                        origin TEXT NOT NULL DEFAULT 'preferences',
                        occurrences INTEGER NOT NULL DEFAULT 1,
                        status TEXT NOT NULL DEFAULT 'pending',
                        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                    )
                    """);
            ensureRuleCandidateColumns(c, st);
            st.executeUpdate("""
                    CREATE TABLE IF NOT EXISTS v2_llm_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        dialog_id TEXT,
                        kind TEXT NOT NULL,
                        request_json TEXT NOT NULL,
                        response_text TEXT NOT NULL DEFAULT '',
                        reasoning_text TEXT NOT NULL DEFAULT '',
                        layer_stats_json TEXT NOT NULL DEFAULT '{}',
                        usage_json TEXT NOT NULL DEFAULT '{}',
                        duration_ms INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL DEFAULT (datetime('now'))
                    )
                    """);
            ensureReasoningColumn(c, st);
            st.executeUpdate("CREATE INDEX IF NOT EXISTS idx_v2_llm_dialog "
                    + "ON v2_llm_log(dialog_id, id)");
            st.executeUpdate("""
                    CREATE TABLE IF NOT EXISTS v2_action_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        dialog_id TEXT,
                        action TEXT NOT NULL,
                        details_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL DEFAULT (datetime('now'))
                    )
                    """);
            st.executeUpdate("CREATE INDEX IF NOT EXISTS idx_v2_action_dialog "
                    + "ON v2_action_log(dialog_id, id)");
            st.executeUpdate("""
                    CREATE TABLE IF NOT EXISTS v2_state (
                        key TEXT PRIMARY KEY,
                        value TEXT
                    )
                    """);
            st.executeUpdate("""
                    CREATE TABLE IF NOT EXISTS v2_profiles (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL UNIQUE,
                        rules_json TEXT NOT NULL,
                        is_builtin INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL DEFAULT (datetime('now')),
                        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                    )
                    """);
            ensureDialogProfileColumns(c, st);
            st.executeUpdate("INSERT OR IGNORE INTO v2_agent_config(id, content) VALUES (1, '{\"rules\":[]}')");
            if (state("lt_initialized") == null) {
                if (agentConfig().path("rules").isEmpty()) saveAgentConfig(defaultAgentConfig().toString());
                setState("lt_initialized", "true");
            }
            insertBuiltinProfiles(c);
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to initialize agent v2 database", e);
        }
    }

    /** Adds reasoning storage to databases created by earlier v2 versions. */
    private static void ensureDialogProfileColumns(Connection c, Statement st) throws SQLException {
        java.util.Set<String> columns = new java.util.HashSet<>();
        try (ResultSet rs = st.executeQuery("PRAGMA table_info(v2_dialogs)")) {
            while (rs.next()) columns.add(rs.getString("name").toLowerCase());
        }
        if (!columns.contains("profile_id")) st.executeUpdate("ALTER TABLE v2_dialogs ADD COLUMN profile_id TEXT NOT NULL DEFAULT ''");
        if (!columns.contains("lt_config")) st.executeUpdate("ALTER TABLE v2_dialogs ADD COLUMN lt_config TEXT NOT NULL DEFAULT ''");
    }

    private static void insertBuiltinProfiles(Connection c) throws SQLException {
        insertBuiltinProfile(c, "brief-plain", "Краткий, без кода", new String[][]{
                {"default-clarity", "behavior", "Отвечай кратко, давая только основной вывод и необходимые пояснения"},
                {"default-code", "output_format", "Не приводи код без явного запроса"},
                {"default-tone", "behavior", "Используй неформальный и дружелюбный тон"}});
        insertBuiltinProfile(c, "detailed-code", "Подробный, с кодом", new String[][]{
                {"default-clarity", "behavior", "Давай подробные структурированные объяснения"},
                {"default-code", "output_format", "Приводи примеры кода, когда они помогают понять решение"},
                {"default-alternatives", "behavior", "Поясняй принципы и альтернативные решения"}});
        insertBuiltinProfile(c, "formal-technical", "Формальный технический", new String[][]{
                {"default-tone", "behavior", "Используй формальный и технически точный стиль"},
                {"default-code", "output_format", "Приводи код только по явному запросу"},
                {"default-evidence", "behavior", "Отделяй факты, предположения и рекомендации"}});
    }

    private static void insertBuiltinProfile(Connection c, String id, String name, String[][] definitions)
            throws SQLException {
        ArrayNode rules = MAPPER.createArrayNode();
        for (String[] definition : definitions) {
            ObjectNode rule = rules.addObject();
            rule.put("id", definition[0]); rule.put("type", definition[1]);
            rule.put("text", definition[2]); rule.put("enabled", true); rule.put("source", "builtin");
        }
        try (PreparedStatement ps = c.prepareStatement(
                "INSERT OR IGNORE INTO v2_profiles(id, name, rules_json, is_builtin) VALUES(?, ?, ?, 1)")) {
            ps.setString(1, id); ps.setString(2, name); ps.setString(3, rules.toString()); ps.executeUpdate();
        }
    }

    private static void ensureRuleCandidateColumns(Connection c, Statement st) throws SQLException {
        java.util.Set<String> columns = new java.util.HashSet<>();
        try (ResultSet rs = st.executeQuery("PRAGMA table_info(v2_rule_candidates)")) {
            while (rs.next()) columns.add(rs.getString("name").toLowerCase());
        }
        for (String column : List.of("operation", "target_rule_id", "base_text", "origin")) {
            if (!columns.contains(column)) {
                String value = "operation".equals(column) ? "'add'"
                        : "origin".equals(column) ? "'preferences'" : "''";
                st.executeUpdate("ALTER TABLE v2_rule_candidates ADD COLUMN " + column
                        + " TEXT NOT NULL DEFAULT " + value);
            }
        }
    }

    private static void ensureReasoningColumn(Connection c, Statement st) throws SQLException {
        boolean exists = false;
        try (ResultSet rs = st.executeQuery("PRAGMA table_info(v2_llm_log)")) {
            while (rs.next()) {
                if ("reasoning_text".equalsIgnoreCase(rs.getString("name"))) {
                    exists = true;
                    break;
                }
            }
        }
        if (!exists) {
            st.executeUpdate("ALTER TABLE v2_llm_log ADD COLUMN reasoning_text TEXT NOT NULL DEFAULT ''");
        }
    }

    // ----- dialogs and transcript -----

    public synchronized void createDialog(String id, String title) {
        createDialog(id, title, "");
    }

    public synchronized void createDialog(String id, String title, String profileId) {
        try (Connection c = connect(); PreparedStatement ps = c.prepareStatement(
                "INSERT OR IGNORE INTO v2_dialogs(id, title, lt_config) VALUES(?, ?, ?)")) {
            ps.setString(1, id);
            ps.setString(2, title == null || title.isBlank() ? "Диалог" : title);
            ps.setString(3, agentConfig().toString());
            ps.executeUpdate();
            if (profileId != null && !profileId.isBlank()) applyProfile(id, profileId);
            setCurrentDialog(id);
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to create v2 dialog " + id, e);
        }
    }

    public synchronized boolean dialogExists(String id) {
        try (Connection c = connect(); PreparedStatement ps = c.prepareStatement(
                "SELECT 1 FROM v2_dialogs WHERE id = ?")) {
            ps.setString(1, id);
            try (ResultSet rs = ps.executeQuery()) {
                return rs.next();
            }
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to check v2 dialog " + id, e);
        }
    }

    public synchronized String dialogTitle(String id) {
        try (Connection c = connect(); PreparedStatement ps = c.prepareStatement(
                "SELECT title FROM v2_dialogs WHERE id = ?")) {
            ps.setString(1, id);
            try (ResultSet rs = ps.executeQuery()) {
                return rs.next() ? rs.getString(1) : "Диалог";
            }
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to read v2 dialog title " + id, e);
        }
    }

    public synchronized List<DialogInfo> listDialogs() {
        List<DialogInfo> result = new ArrayList<>();
        try (Connection c = connect(); PreparedStatement ps = c.prepareStatement(
                "SELECT id, title, created_at, updated_at FROM v2_dialogs ORDER BY updated_at DESC");
             ResultSet rs = ps.executeQuery()) {
            while (rs.next()) {
                result.add(new DialogInfo(rs.getString(1), rs.getString(2), rs.getString(3), rs.getString(4)));
            }
            return result;
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to list v2 dialogs", e);
        }
    }

    public synchronized String currentDialogId() {
        return state(CURRENT_DIALOG);
    }

    public synchronized void setCurrentDialog(String id) {
        setState(CURRENT_DIALOG, id);
    }

    public synchronized void appendEntry(String dialogId, String kind, String payload) {
        try (Connection c = connect()) {
            try (PreparedStatement ps = c.prepareStatement(
                    "INSERT INTO v2_entries(dialog_id, kind, payload) VALUES(?, ?, ?)")) {
                ps.setString(1, dialogId);
                ps.setString(2, kind);
                ps.setString(3, payload);
                ps.executeUpdate();
            }
            try (PreparedStatement ps = c.prepareStatement(
                    "UPDATE v2_dialogs SET updated_at = datetime('now') WHERE id = ?")) {
                ps.setString(1, dialogId);
                ps.executeUpdate();
            }
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to append v2 entry for " + dialogId, e);
        }
    }

    public synchronized List<StoredEntry> loadEntries(String dialogId, String kind) {
        return loadEntries(dialogId, kind, false);
    }

    public synchronized List<StoredEntry> loadEntriesExcept(String dialogId, String excludedKind) {
        return loadEntries(dialogId, excludedKind, true);
    }

    private List<StoredEntry> loadEntries(String dialogId, String kind, boolean exclude) {
        List<StoredEntry> result = new ArrayList<>();
        String sql = "SELECT kind, payload FROM v2_entries WHERE dialog_id = ? AND kind "
                + (exclude ? "<> ?" : "= ?") + " ORDER BY id";
        try (Connection c = connect(); PreparedStatement ps = c.prepareStatement(sql)) {
            ps.setString(1, dialogId);
            ps.setString(2, kind);
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    result.add(new StoredEntry(rs.getString(1), rs.getString(2)));
                }
            }
            return result;
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to load v2 entries for " + dialogId, e);
        }
    }

    // ----- short-term sticky facts -----

    private static String normalizeFactKey(String key) {
        return key.toLowerCase().replace('ё', 'е').replaceAll("\\s+", " ").trim();
    }

    public synchronized List<AgentProtocol.Fact> listFacts(String dialogId) {
        List<AgentProtocol.Fact> result = new ArrayList<>();
        try (Connection c = connect(); PreparedStatement ps = c.prepareStatement(
                "SELECT category, key_display, value FROM v2_facts WHERE dialog_id = ? ORDER BY updated_at, id")) {
            ps.setString(1, dialogId);
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    result.add(new AgentProtocol.Fact(rs.getString(1), rs.getString(2), rs.getString(3)));
                }
            }
            return result;
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to load v2 facts for " + dialogId, e);
        }
    }

    public synchronized List<AgentProtocol.Fact> upsertFacts(String dialogId,
            List<AgentProtocol.Fact> facts, int maxFacts) {
        try (Connection c = connect(); PreparedStatement ps = c.prepareStatement("""
                INSERT INTO v2_facts(dialog_id, category, key_norm, key_display, value)
                VALUES(?, ?, ?, ?, ?)
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
            throw new IllegalStateException("Failed to upsert v2 facts for " + dialogId, e);
        }
        if (maxFacts > 0) {
            try (Connection c = connect(); PreparedStatement ps = c.prepareStatement("""
                    DELETE FROM v2_facts WHERE dialog_id = ? AND id NOT IN (
                        SELECT id FROM v2_facts WHERE dialog_id = ?
                        ORDER BY updated_at DESC, id DESC LIMIT ?)
                    """)) {
                ps.setString(1, dialogId);
                ps.setString(2, dialogId);
                ps.setInt(3, maxFacts);
                ps.executeUpdate();
            } catch (SQLException e) {
                throw new IllegalStateException("Failed to limit v2 facts for " + dialogId, e);
            }
        }
        return listFacts(dialogId);
    }

    public synchronized List<AgentProtocol.Fact> replaceFacts(String dialogId,
            List<AgentProtocol.Fact> facts) {
        clearFacts(dialogId);
        return upsertFacts(dialogId, facts, 0);
    }

    public synchronized void clearFacts(String dialogId) {
        executeUpdate("DELETE FROM v2_facts WHERE dialog_id = ?", dialogId);
    }

    // ----- working memory -----

    public synchronized ObjectNode working(String dialogId) {
        ObjectNode result = MAPPER.createObjectNode();
        try (Connection c = connect(); PreparedStatement ps = c.prepareStatement(
                "SELECT field, value FROM v2_working WHERE dialog_id = ? ORDER BY field")) {
            ps.setString(1, dialogId);
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    String raw = rs.getString(2);
                    try {
                        result.set(rs.getString(1), MAPPER.readTree(raw));
                    } catch (Exception e) {
                        result.put(rs.getString(1), raw);
                    }
                }
            }
            return result;
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to load v2 working memory for " + dialogId, e);
        }
    }

    public synchronized void saveWorking(String dialogId, JsonNode working) {
        if (working == null || !working.isObject()) {
            throw new IllegalArgumentException("Working memory must be a JSON object");
        }
        try (Connection c = connect()) {
            c.setAutoCommit(false);
            try (PreparedStatement delete = c.prepareStatement(
                    "DELETE FROM v2_working WHERE dialog_id = ?");
                 PreparedStatement insert = c.prepareStatement("""
                         INSERT INTO v2_working(dialog_id, field, value) VALUES(?, ?, ?)
                         ON CONFLICT(dialog_id, field) DO UPDATE SET
                             value = excluded.value, updated_at = datetime('now')
                         """)) {
                delete.setString(1, dialogId);
                delete.executeUpdate();
                Iterator<String> fields = working.fieldNames();
                while (fields.hasNext()) {
                    String field = fields.next();
                    JsonNode value = working.get(field);
                    insert.setString(1, dialogId);
                    insert.setString(2, field);
                    insert.setString(3, value.isTextual() ? value.asText() : value.toString());
                    insert.addBatch();
                }
                insert.executeBatch();
                c.commit();
            } catch (Exception e) {
                c.rollback();
                throw e;
            } finally {
                c.setAutoCommit(true);
            }
        } catch (Exception e) {
            throw new IllegalStateException("Failed to save v2 working memory for " + dialogId, e);
        }
    }

    public synchronized void clearWorking(String dialogId) {
        executeUpdate("DELETE FROM v2_working WHERE dialog_id = ?", dialogId);
    }

    // ----- long-term agent config -----

    public static ObjectNode defaultAgentConfig() {
        ObjectNode config = MAPPER.createObjectNode();
        ArrayNode rules = config.putArray("rules");
        addDefaultRule(rules, "default-language", "Отвечай на языке пользователя");
        addDefaultRule(rules, "default-honesty", "Не выдумывай неизвестные сведения");
        addDefaultRule(rules, "default-format", "Используй Markdown только когда он улучшает читаемость");
        addDefaultRule(rules, "default-clarity", "Сохраняй краткость без потери существенной информации");
        return config;
    }

    private static void addDefaultRule(ArrayNode rules, String id, String text) {
        ObjectNode rule = rules.addObject();
        rule.put("id", id);
        rule.put("type", "behavior");
        rule.put("text", text);
        rule.put("enabled", true);
        rule.put("source", "builtin");
    }

    public synchronized ObjectNode agentConfig() {
        try (Connection c = connect(); PreparedStatement ps = c.prepareStatement(
                "SELECT content FROM v2_agent_config WHERE id = 1");
             ResultSet rs = ps.executeQuery()) {
            String raw = rs.next() ? rs.getString(1) : "{\"rules\":[]}";
            JsonNode parsed = MAPPER.readTree(raw);
            return parsed != null && parsed.isObject() ? (ObjectNode) parsed : MAPPER.createObjectNode();
        } catch (Exception e) {
            throw new IllegalStateException("Failed to load v2 agent config", e);
        }
    }

    public synchronized void saveAgentConfig(String content) {
        executeUpdate("UPDATE v2_agent_config SET content = ?, updated_at = datetime('now') WHERE id = 1", content);
    }

    public synchronized ObjectNode dialogAgentConfig(String dialogId) {
        String raw;
        try (Connection c = connect(); PreparedStatement ps = c.prepareStatement(
                "SELECT lt_config FROM v2_dialogs WHERE id = ?")) {
            ps.setString(1, dialogId);
            try (ResultSet rs = ps.executeQuery()) {
                raw = rs.next() ? rs.getString(1) : "";
            }
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to load dialog LT config " + dialogId, e);
        }
        try {
            if (raw == null || raw.isBlank()) {
                ObjectNode inherited = agentConfig();
                saveDialogAgentConfig(dialogId, inherited.toString());
                return inherited;
            }
            JsonNode parsed = MAPPER.readTree(raw);
            if (parsed == null || !parsed.isObject()) throw new IllegalArgumentException("Invalid dialog LT config");
            return (ObjectNode) parsed;
        } catch (Exception e) {
            throw new IllegalStateException("Failed to load dialog LT config " + dialogId, e);
        }
    }

    public synchronized void saveDialogAgentConfig(String dialogId, String content) {
        executeUpdate("UPDATE v2_dialogs SET lt_config = ?, updated_at = datetime('now') WHERE id = ?", content, dialogId);
    }

    public synchronized ArrayNode profiles() {
        ArrayNode result = MAPPER.createArrayNode();
        try (Connection c = connect(); Statement st = c.createStatement(); ResultSet rs = st.executeQuery(
                "SELECT id, name, rules_json, is_builtin FROM v2_profiles ORDER BY is_builtin DESC, name")) {
            while (rs.next()) {
                ObjectNode item = result.addObject();
                item.put("id", rs.getString(1)); item.put("name", rs.getString(2));
                item.set("rules", readJson(rs.getString(3))); item.put("builtin", rs.getInt(4) == 1);
            }
            return result;
        } catch (SQLException e) { throw new IllegalStateException("Failed to load profiles", e); }
    }

    public synchronized ObjectNode profile(String id) {
        for (JsonNode item : profiles()) if (id.equals(item.path("id").asText())) return (ObjectNode) item;
        return null;
    }

    public synchronized void saveProfile(String id, String name, String rulesJson, boolean builtin) {
        executeUpdate("""
                INSERT INTO v2_profiles(id, name, rules_json, is_builtin) VALUES(?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET name = excluded.name, rules_json = excluded.rules_json,
                    updated_at = datetime('now')
                """, id, name, rulesJson, builtin ? "1" : "0");
    }

    public synchronized void deleteProfile(String id) { executeUpdate("DELETE FROM v2_profiles WHERE id = ? AND is_builtin = 0", id); }

    public synchronized void applyProfile(String dialogId, String profileId) {
        ObjectNode profile = profile(profileId);
        if (profile == null) throw new IllegalArgumentException("Profile not found");
        executeUpdate("UPDATE v2_dialogs SET profile_id = ?, lt_config = ?, updated_at = datetime('now') WHERE id = ?",
                profileId, MAPPER.createObjectNode().set("rules", profile.get("rules")).toString(), dialogId);
    }

    public synchronized String dialogProfileId(String dialogId) {
        try (Connection c = connect(); PreparedStatement ps = c.prepareStatement("SELECT profile_id FROM v2_dialogs WHERE id = ?")) {
            ps.setString(1, dialogId); try (ResultSet rs = ps.executeQuery()) { return rs.next() ? rs.getString(1) : ""; }
        } catch (SQLException e) { throw new IllegalStateException("Failed to load dialog profile", e); }
    }

    /** Records another observation of a preference; approved/rejected keys never reappear. */
    public synchronized void observeRuleCandidate(String key, String text, String sourceFact,
                                                   String dialogId, String operation,
                                                   String targetRuleId, String baseText, String origin) {
        String scopedKey = (dialogId == null || dialogId.isBlank() ? "global" : dialogId) + ":" + key;
        executeUpdate("""
                INSERT INTO v2_rule_candidates(candidate_key, rule_text, source_fact, source_dialog,
                    operation, target_rule_id, base_text, origin)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_key) DO UPDATE SET
                    rule_text = CASE WHEN v2_rule_candidates.status = 'pending'
                        THEN excluded.rule_text ELSE v2_rule_candidates.rule_text END,
                    source_fact = excluded.source_fact,
                    source_dialog = excluded.source_dialog,
                    operation = excluded.operation,
                    target_rule_id = excluded.target_rule_id,
                    base_text = excluded.base_text,
                    origin = excluded.origin,
                    occurrences = CASE WHEN v2_rule_candidates.status = 'pending'
                        THEN v2_rule_candidates.occurrences + 1 ELSE v2_rule_candidates.occurrences END,
                    updated_at = datetime('now')
                """, scopedKey, text, sourceFact, dialogId, operation, targetRuleId, baseText, origin);
    }

    public synchronized ArrayNode ruleCandidates(boolean readyOnly) {
        return ruleCandidates(readyOnly, null);
    }

    public synchronized ArrayNode ruleCandidates(boolean readyOnly, String dialogId) {
        ArrayNode result = MAPPER.createArrayNode();
        String sql = "SELECT id, candidate_key, rule_text, source_fact, source_dialog, occurrences, "
                + "operation, target_rule_id, base_text, origin FROM v2_rule_candidates WHERE status = 'pending'"
                + (readyOnly ? " AND (occurrences >= 2 OR origin = 'bootstrap')" : "")
                + (dialogId == null ? "" : " AND source_dialog = ?") + " ORDER BY updated_at DESC";
        try (Connection c = connect(); PreparedStatement ps = c.prepareStatement(sql)) {
            if (dialogId != null) ps.setString(1, dialogId);
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    ObjectNode item = result.addObject();
                    item.put("id", rs.getLong(1));
                    item.put("key", rs.getString(2));
                    item.put("text", rs.getString(3));
                    item.put("sourceFact", rs.getString(4));
                    item.put("sourceDialog", rs.getString(5));
                    item.put("occurrences", rs.getInt(6));
                    item.put("operation", rs.getString(7));
                    item.put("targetRuleId", rs.getString(8));
                    item.put("baseText", rs.getString(9));
                    item.put("origin", rs.getString(10));
                }
                return result;
            }
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to load LT rule candidates", e);
        }
    }

    public synchronized ObjectNode ruleCandidate(long id) {
        try (Connection c = connect(); PreparedStatement ps = c.prepareStatement(
                "SELECT id, candidate_key, rule_text, source_fact, source_dialog, occurrences, status, "
                        + "operation, target_rule_id, base_text, origin FROM v2_rule_candidates WHERE id = ?")) {
            ps.setLong(1, id);
            try (ResultSet rs = ps.executeQuery()) {
                if (!rs.next()) return null;
                ObjectNode item = MAPPER.createObjectNode();
                item.put("id", rs.getLong(1));
                item.put("key", rs.getString(2));
                item.put("text", rs.getString(3));
                item.put("sourceFact", rs.getString(4));
                item.put("sourceDialog", rs.getString(5));
                item.put("occurrences", rs.getInt(6));
                item.put("status", rs.getString(7));
                item.put("operation", rs.getString(8));
                item.put("targetRuleId", rs.getString(9));
                item.put("baseText", rs.getString(10));
                item.put("origin", rs.getString(11));
                return item;
            }
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to load LT rule candidate " + id, e);
        }
    }

    public synchronized void resolveRuleCandidate(long id, String status) {
        executeUpdate("UPDATE v2_rule_candidates SET status = ?, updated_at = datetime('now') WHERE id = ?",
                status, String.valueOf(id));
    }

    // ----- logs -----

    public synchronized long logLlm(String dialogId, String kind, String requestJson,
            String responseText, String reasoningText, String layerStatsJson,
            String usageJson, long durationMs) {
        try (Connection c = connect(); PreparedStatement ps = c.prepareStatement("""
                INSERT INTO v2_llm_log(dialog_id, kind, request_json, response_text,
                    reasoning_text, layer_stats_json, usage_json, duration_ms)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """)) {
            ps.setString(1, dialogId);
            ps.setString(2, kind);
            ps.setString(3, requestJson);
            ps.setString(4, responseText == null ? "" : responseText);
            ps.setString(5, reasoningText == null ? "" : reasoningText);
            ps.setString(6, layerStatsJson == null ? "{}" : layerStatsJson);
            ps.setString(7, usageJson == null ? "{}" : usageJson);
            ps.setLong(8, durationMs);
            ps.executeUpdate();
            try (Statement st = c.createStatement(); ResultSet rs = st.executeQuery("SELECT last_insert_rowid()")) {
                rs.next();
                return rs.getLong(1);
            }
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to write v2 LLM log", e);
        }
    }

    public synchronized void logAction(String dialogId, String action, String detailsJson) {
        try (Connection c = connect(); PreparedStatement ps = c.prepareStatement(
                "INSERT INTO v2_action_log(dialog_id, action, details_json) VALUES(?, ?, ?)")) {
            ps.setString(1, dialogId);
            ps.setString(2, action);
            ps.setString(3, detailsJson == null ? "{}" : detailsJson);
            ps.executeUpdate();
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to write v2 action log", e);
        }
    }

    public synchronized ArrayNode llmLogs(String dialogId) {
        ArrayNode result = MAPPER.createArrayNode();
        String sql = "SELECT id, dialog_id, kind, request_json, response_text, reasoning_text, "
                + "layer_stats_json, usage_json, duration_ms, created_at FROM v2_llm_log "
                + (dialogId == null ? "ORDER BY id DESC LIMIT 500" : "WHERE dialog_id = ? ORDER BY id DESC");
        try (Connection c = connect(); PreparedStatement ps = c.prepareStatement(sql)) {
            if (dialogId != null) {
                ps.setString(1, dialogId);
            }
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    ObjectNode item = result.addObject();
                    item.put("id", rs.getLong(1));
                    item.put("dialogId", rs.getString(2));
                    item.put("kind", rs.getString(3));
                    item.set("request", readJson(rs.getString(4)));
                    item.put("response", rs.getString(5));
                    item.put("reasoning", rs.getString(6));
                    item.set("layers", readJson(rs.getString(7)));
                    item.set("usage", readJson(rs.getString(8)));
                    item.put("durationMs", rs.getLong(9));
                    item.put("createdAt", rs.getString(10));
                }
            }
            return result;
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to read v2 LLM logs", e);
        }
    }

    public synchronized ArrayNode actionLogs(String dialogId) {
        ArrayNode result = MAPPER.createArrayNode();
        String sql = "SELECT id, dialog_id, action, details_json, created_at FROM v2_action_log "
                + (dialogId == null ? "ORDER BY id DESC LIMIT 500" : "WHERE dialog_id = ? ORDER BY id DESC LIMIT 500");
        try (Connection c = connect(); PreparedStatement ps = c.prepareStatement(sql)) {
            if (dialogId != null) {
                ps.setString(1, dialogId);
            }
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    ObjectNode item = result.addObject();
                    item.put("id", rs.getLong(1));
                    item.put("dialogId", rs.getString(2));
                    item.put("action", rs.getString(3));
                    item.set("details", readJson(rs.getString(4)));
                    item.put("createdAt", rs.getString(5));
                }
            }
            return result;
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to read v2 action logs", e);
        }
    }

    // ----- clearing and state -----

    /** Clears operational ST while preserving the visible transcript and logs. */
    public synchronized void clearShortTerm(String dialogId) {
        try (Connection c = connect()) {
            c.setAutoCommit(false);
            try {
                try (PreparedStatement ps = c.prepareStatement(
                        "DELETE FROM v2_entries WHERE dialog_id = ? AND kind IN ('history', 'summary', 'facts')")) {
                    ps.setString(1, dialogId);
                    ps.executeUpdate();
                }
                try (PreparedStatement ps = c.prepareStatement("DELETE FROM v2_facts WHERE dialog_id = ?")) {
                    ps.setString(1, dialogId);
                    ps.executeUpdate();
                }
                c.commit();
            } catch (Exception e) {
                c.rollback();
                throw e;
            } finally {
                c.setAutoCommit(true);
            }
        } catch (Exception e) {
            throw new IllegalStateException("Failed to clear v2 short-term memory for " + dialogId, e);
        }
    }

    public synchronized void resetAll() {
        try (Connection c = connect(); Statement st = c.createStatement()) {
            c.setAutoCommit(false);
            try {
                st.executeUpdate("DELETE FROM v2_entries WHERE kind IN ('history', 'summary', 'facts')");
                st.executeUpdate("DELETE FROM v2_facts");
                st.executeUpdate("DELETE FROM v2_working");
                st.executeUpdate("DELETE FROM v2_rule_candidates");
                try (PreparedStatement ps = c.prepareStatement(
                        "UPDATE v2_dialogs SET lt_config = ?, profile_id = ''")) {
                    ps.setString(1, defaultAgentConfig().toString());
                    ps.executeUpdate();
                }
                try (PreparedStatement ps = c.prepareStatement(
                        "UPDATE v2_agent_config SET content = ?, updated_at = datetime('now') WHERE id = 1")) {
                    ps.setString(1, defaultAgentConfig().toString());
                    ps.executeUpdate();
                }
                c.commit();
            } catch (Exception e) {
                c.rollback();
                throw e;
            } finally {
                c.setAutoCommit(true);
            }
            setState("lt_initialized", "true");
            executeUpdate("DELETE FROM v2_state WHERE key = 'lt_bootstrap_completed' OR key LIKE 'lt_bootstrap_completed:%'");
            logAction(null, "reset_all", "{}");
        } catch (Exception e) {
            throw new IllegalStateException("Failed to reset v2 memory", e);
        }
    }

    public synchronized String state(String key) {
        try (Connection c = connect(); PreparedStatement ps = c.prepareStatement(
                "SELECT value FROM v2_state WHERE key = ?")) {
            ps.setString(1, key);
            try (ResultSet rs = ps.executeQuery()) {
                return rs.next() ? rs.getString(1) : null;
            }
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to read v2 state " + key, e);
        }
    }

    public synchronized void setState(String key, String value) {
        try (Connection c = connect(); PreparedStatement ps = c.prepareStatement(
                "INSERT INTO v2_state(key, value) VALUES(?, ?) "
                        + "ON CONFLICT(key) DO UPDATE SET value = excluded.value")) {
            ps.setString(1, key);
            ps.setString(2, value);
            ps.executeUpdate();
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to save v2 state " + key, e);
        }
    }

    private void executeUpdate(String sql, String... args) {
        try (Connection c = connect(); PreparedStatement ps = c.prepareStatement(sql)) {
            for (int i = 0; i < args.length; i++) {
                ps.setString(i + 1, args[i]);
            }
            ps.executeUpdate();
        } catch (SQLException e) {
            throw new IllegalStateException("Failed v2 database update", e);
        }
    }

    private static JsonNode readJson(String raw) {
        try {
            return MAPPER.readTree(raw);
        } catch (Exception e) {
            return MAPPER.createObjectNode().put("raw", raw);
        }
    }
}
