package ru.tionstudio.ai.advent.challenge.agent.v2;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.EnumMap;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

/**
 * Code-enforced task lifecycle. The model may request a transition, but only
 * this class decides whether it is legal; a finished task is immutable.
 */
public final class TaskStateMachine {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    public enum Stage {
        RESEARCH, PLANNING, EXECUTION, VALIDATION, DONE;

        public static Stage parse(String value) {
            if (value == null) return null;
            try {
                return valueOf(value.trim().toUpperCase());
            } catch (IllegalArgumentException e) {
                return null;
            }
        }

        public String wireName() {
            return name().toLowerCase();
        }
    }

    private static final Map<Stage, Set<Stage>> TRANSITIONS = new EnumMap<>(Stage.class);

    static {
        TRANSITIONS.put(Stage.RESEARCH, Set.of(Stage.PLANNING));
        TRANSITIONS.put(Stage.PLANNING, Set.of(Stage.RESEARCH, Stage.EXECUTION));
        TRANSITIONS.put(Stage.EXECUTION, Set.of(Stage.PLANNING, Stage.VALIDATION));
        TRANSITIONS.put(Stage.VALIDATION, Set.of(Stage.EXECUTION, Stage.DONE));
        TRANSITIONS.put(Stage.DONE, Set.of());
    }

    public record Decision(boolean allowed, String reason) { }

    private TaskStateMachine() { }

    public static ObjectNode newTask(String task, String previousTask) {
        ObjectNode state = MAPPER.createObjectNode();
        state.put("taskId", UUID.randomUUID().toString());
        state.put("task", task == null ? "" : task.trim());
        state.put("stage", Stage.PLANNING.wireName());
        state.put("step", "Составить план");
        state.put("expected", "План выполнения задачи");
        state.put("planStatus", "draft");
        state.put("validationStatus", "not_started");
        state.put("validationFeedback", "");
        state.put("paused", false);
        state.put("waitingForUser", false);
        state.put("pendingQuestion", "");
        state.putArray("pendingOptions");
        state.put("pendingAllowCustom", true);
        state.put("revision", 1);
        if (previousTask != null && !previousTask.isBlank()) {
            state.put("previousTask", previousTask);
        }
        state.putArray("plan");
        state.put("currentStep", 0);
        return state;
    }

    public static Decision check(JsonNode state, Stage target) {
        if (target == null) {
            return new Decision(false, "Неизвестный целевой этап. Допустимые: research, planning, execution, validation, done.");
        }
        Stage current = Stage.parse(state.path("stage").asText(""));
        if (current == null) {
            return new Decision(false, "Текущее состояние задачи повреждено: неизвестный этап.");
        }
        if (current == Stage.DONE) {
            return new Decision(false, "Завершённая задача неизменяема. Новое сообщение пользователя создаёт новую задачу.");
        }
        if (current == target) {
            return new Decision(false, "Задача уже находится на этапе " + current.wireName() + ".");
        }
        if (!TRANSITIONS.get(current).contains(target)) {
            return new Decision(false, "Переход " + current.wireName() + " → " + target.wireName()
                    + " запрещён. Разрешено: " + allowedNames(current) + ".");
        }
        if (current == Stage.PLANNING && target == Stage.EXECUTION
                && !"approved".equals(state.path("planStatus").asText())) {
            return new Decision(false, "Нельзя начать execution до утверждения плана.");
        }
        if (current == Stage.EXECUTION && target == Stage.VALIDATION && !validationReady(state)) {
            return new Decision(false, "Нельзя начать финальную валидацию: все обязательные шаги должны быть фактически выполнены.");
        }
        if (current == Stage.VALIDATION && target == Stage.DONE) {
            if (!"passed".equals(state.path("validationStatus").asText())) {
                return new Decision(false, "Нельзя завершить задачу без успешной отдельной валидации.");
            }
            if (!validationReady(state)) {
                return new Decision(false, "Нельзя завершить задачу: обязательные шаги не выполнены.");
            }
        }
        return new Decision(true, "Переход разрешён");
    }

    /** All steps are processed; optional steps may be skipped. */
    public static boolean planComplete(JsonNode state) {
        JsonNode plan = state.path("plan");
        if (!plan.isArray() || plan.isEmpty()) return false;
        for (JsonNode step : plan) {
            String status = step.path("status").asText("pending");
            if (!Set.of("done", "skipped").contains(status)) return false;
        }
        return true;
    }

    /** Validation requires real results for required steps, not just skipped statuses. */
    public static boolean validationReady(JsonNode state) {
        if (!planComplete(state)) return false;
        boolean hasCompletedRequiredStep = false;
        for (JsonNode step : state.path("plan")) {
            boolean required = step.path("requiredForValidation").asBoolean(true);
            if (required && !"done".equals(step.path("status").asText())) return false;
            if (required && "done".equals(step.path("status").asText())
                    && !step.path("result").asText("").isBlank()) {
                hasCompletedRequiredStep = true;
            }
        }
        return hasCompletedRequiredStep;
    }

    /** Applies an already approved transition and increments the revision. */
    public static ObjectNode apply(JsonNode source, Stage target, String step, String expected) {
        ObjectNode next = source.deepCopy();
        next.put("stage", target.wireName());
        if (step != null && !step.isBlank()) next.put("step", step.trim());
        if (expected != null && !expected.isBlank()) next.put("expected", expected.trim());
        next.put("revision", source.path("revision").asInt(0) + 1);
        return next;
    }

    public static String allowedNames(Stage stage) {
        Set<Stage> targets = TRANSITIONS.get(stage);
        if (targets == null || targets.isEmpty()) return "нет";
        return targets.stream().map(Stage::wireName).sorted().reduce((a, b) -> a + ", " + b).orElse("нет");
    }
}
