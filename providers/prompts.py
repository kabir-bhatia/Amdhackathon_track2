"""Prompt templates shared by all LLM providers (v1).

Tuned for the two LLM-Judge axes:
  - caption accuracy: keep the concrete facts from the vision description.
  - style match: commit hard to the requested tone.
"""

VISION_PROMPT = (
    "Describe what happens in this video in 2-3 plain, factual sentences. "
    "Name the main subjects, setting, actions, and notable visual details "
    "(colors, weather, objects). Do not add opinions, jokes, or flourishes."
)

# Per-style instruction appended to the styling prompt.
STYLE_INSTRUCTIONS = {
    "formal": (
        "Write a single professional, objective caption. Neutral factual tone, "
        "no humor, no slang. One or two sentences."
    ),
    "sarcastic": (
        "Write a single dry, ironic, lightly mocking caption. Deadpan wit, "
        "gentle eye-roll energy, but still describe what actually happens. One or two sentences."
    ),
    "humorous_tech": (
        "Write a single funny caption that lands a technology or programming joke "
        "(bugs, deploys, APIs, RAM, git, etc.) tied to what happens on screen. One or two sentences."
    ),
    "humorous_non_tech": (
        "Write a single funny, everyday caption with light relatable humor and "
        "absolutely no technical or programming references. One or two sentences."
    ),
}


def build_style_prompt(description: str, style: str) -> str:
    instruction = STYLE_INSTRUCTIONS[style]
    return (
        f"Video description:\n{description}\n\n"
        f"Task: {instruction}\n"
        "Stay faithful to the described content. Do NOT think out loud, brainstorm, "
        "list options, or explain — output ONLY the final caption itself, at most 30 words, "
        "no labels, no quotes, no preamble."
    )
