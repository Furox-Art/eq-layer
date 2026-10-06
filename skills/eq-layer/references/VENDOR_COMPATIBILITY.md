# Vendor compatibility

Verified against public documentation on 2026-10-06.

| Ecosystem | Project-level discovery path in this repository |
| --- | --- |
| OpenAI Codex | `.codex/skills/eq-layer/SKILL.md` |
| Anthropic Claude Code | `.claude/skills/eq-layer/SKILL.md` |
| GitHub Copilot | `.github/skills/eq-layer/SKILL.md` (also accepts `.agents/skills`) |
| Cursor | `.cursor/skills/eq-layer/SKILL.md` (also accepts `.agents/skills`) |
| Google Antigravity CLI | `.agents/skills/eq-layer/SKILL.md` |
| xAI Grok Build | `.grok/skills/eq-layer/SKILL.md` |
| Portable Agent Plugins clients | `skills/eq-layer/SKILL.md` via root `plugin.json` |

The canonical implementation lives under `skills/eq-layer/`. Vendor-specific
project skill files are discovery shims so behavior is maintained in one place.

## User-level installation

```bash
python tools/install_agent_skills.py --target all
```

The installer copies the canonical skill into the user-level discovery roots
for Codex, Claude Code, GitHub Copilot, Cursor, Grok, and the common
`~/.agents/skills` location. It does not overwrite an existing installation
unless `--force` is passed.

## Official references

- OpenAI: https://developers.openai.com/api/docs/guides/tools-skills
- OpenAI Codex: https://developers.openai.com/blog/eval-skills
- GitHub Copilot: https://docs.github.com/en/copilot/how-tos/copilot-on-github/customize-copilot/customize-cloud-agent/add-skills
- Cursor: https://cursor.com/docs/skills
- Google Antigravity: https://codelabs.developers.google.com/antigravity/how-to-create-agent-skills-for-antigravity-cli
- xAI Grok Build: https://docs.x.ai/build/features/skills-plugins-marketplaces
- Agent Plugins 1.0: https://agent-plugins.org/specification
