package ru.tionstudio.ai.advent.challenge.agent.v2;

/** Cheap prompt-size estimate for providers without a local tokenizer. */
public final class TokenEstimator {
    private TokenEstimator() { }

    public static int estimate(String text, double calibration) {
        if (text == null || text.isEmpty()) {
            return 0;
        }
        double factor = calibration > 0 ? calibration : 1.0;
        return Math.max(1, (int) Math.ceil((text.length() / 3.0) * factor));
    }
}
