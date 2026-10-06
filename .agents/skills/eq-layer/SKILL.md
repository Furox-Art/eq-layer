---
name: eq-layer
description: Apply EQ-Layer's inspectable intent, affect, repair, interaction-quality, dialogue-reference, and factored-action control pass before drafting a response.
license: MIT
---

# EQ-Layer compatibility shim

The authoritative skill is `skills/eq-layer/SKILL.md` at the repository root.
Read that file before using this skill and follow it as the canonical workflow.
Use `skills/eq-layer/scripts/route.py` for executable routing. Do not invent a
router result when the script cannot run; use the canonical skill's conservative
manual fallback instead.
