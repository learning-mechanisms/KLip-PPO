"""
Utility modules shared across the codebase.

Nothing in ``klip_ppo.core`` should hard-code a path, a seed routine, or a git
invocation; everything lives here. ``core`` may import from ``utils``; ``utils`` must
not import from ``core``.
"""
