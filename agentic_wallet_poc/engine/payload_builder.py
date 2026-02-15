"""
Payload builder — thin wrapper over PlaybookEngine for backward compatibility.

All protocol-specific logic now lives in JSON playbook files under data/playbooks/.
This module preserves the old convert_human_to_payload() signature so that existing
callers (annotate_with_hybrid.py, evaluation scripts) continue to work.
"""

from typing import Dict, Any, Optional

from engine.playbook_engine import PlaybookEngine

# Module-level singleton (lazy-initialized)
_engine: Optional[PlaybookEngine] = None


def _get_engine(token_resolver, ens_resolver) -> PlaybookEngine:
    """Get or create a PlaybookEngine singleton."""
    global _engine
    if _engine is None:
        _engine = PlaybookEngine(
            token_resolver=token_resolver,
            ens_resolver=ens_resolver,
        )
    return _engine


def convert_human_to_payload(
    llm_payload: Dict[str, Any],
    token_resolver,
    ens_resolver,
    chain_id: int = 1,
    from_address: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Convert LLM output (possibly human-readable) to final ExecutablePayload dict.

    This is a backward-compatible wrapper around PlaybookEngine.build_payload().
    All protocol knowledge comes from playbook JSON files.
    """
    engine = _get_engine(token_resolver, ens_resolver)
    return engine.build_payload(llm_payload, chain_id=chain_id, from_address=from_address)
