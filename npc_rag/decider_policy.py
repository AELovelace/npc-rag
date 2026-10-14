"""Shared, dependency-free routing schema for native inference and adapter training."""

DECIDER_POLICY = "game-chat-v3"
DECIDER_QUESTION = "Is the player asking for game help or information, or having a social or off-topic conversation?"
DECIDER_OPTIONS = (
    "Game help or info: how to play, rules, controls, items, places, lore, quests or a gameplay problem",
    "Social or off-topic: feelings, jokes, preferences, chatting with the character, or real-world questions",
)
DECIDER_LABELS = dict(zip(DECIDER_OPTIONS, ("game", "general")))
