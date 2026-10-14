# GAME / CHAT labeling guide (game-chat-v3)

Label the player's intent, not punctuation, spelling, politeness or the presence of a game noun. `general` is the stored label for CHAT.

- **GAME (`game_help`)**: requests for game facts, rules, controls, items, places, lore, NPC roles, quests, or advice about what to do next. Terse keyword lookups count as requests. Questions about real-world subjects are CHAT.
- **GAME (`game_problem`)**: a current gameplay obstacle, bug report, depleted resource or repeated unwanted mechanical outcome implying a need for help, even without a question. A bare emotional reaction is insufficient.
- **CHAT (`character_opinion`)**: asks the character's personal taste or feelings about a game place, item or event. Asking which choice improves a game outcome is GAME instead.
- **CHAT (`vent_only`)**: recounts a completed event or expresses frustration without asking for information or describing an unresolved obstacle. For example, disappointment after a lost tournament is CHAT; being unable to claim a tournament reward is GAME.
- **GAME (`mixed_game_request`)**: a greeting, emotional preamble or thanks attached to any game information/help request. The actionable request takes priority.
- **CHAT (`off_topic`)**: real-world trivia, maths, dates, everyday life, AI identity questions, jokes, roleplay gestures and social conversation.
- **CHAT (`context_missing`)**: an isolated acknowledgement, reaction, emoji or deictic fragment without a recoverable request. This message-only policy cannot resolve references to prior dialogue; do not invent context.

Every synthetic row records its rule, family and provenance. All rows remain drafts pending independent review. Keep semantic pairs and paraphrases in the same partition. Do not count prefixes, punctuation changes or spelling noise as new authored examples.

The realistic development set is separately authored and is used for checkpoint selection by worst-class recall (then accuracy, then loss). Synthetic validation is secondary. The fixed `eval_messages.json` and `eval_mixed_messages.json` files are exclusion checks and final qualification only: never copy their wording, train on them, calibrate on them or select a checkpoint using them. A disappointing final score must be reported, not used to start another selection loop.

Augmentation is sampled anew per training epoch with the same probabilities for both labels. Record applied operations in `variant`; never augment development, validation or final test inputs. Temperature calibration, if performed, uses development data only and is saved with the adapter. It does not change argmax labels.
