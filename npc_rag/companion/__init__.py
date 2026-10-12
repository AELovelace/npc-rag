"""Companion characters (Dame Elowen, Mira, ...): a LangGraph pipeline with backstories, memories and gossip.

npc-rag stays stateless for companions: the game server owns every memory and sends the candidates with each
request; the only thing kept here is the backstory (lore) the game server pushes, plus caches.
"""
