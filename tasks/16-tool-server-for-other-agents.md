# 16 · dMLX as a tool server for other local agents

**Status:** Open (from the competitor comparison, 2026-11-02).

## Why
Ollama (images, editing "future") and LLMTray (chat with an edit tool) are moving into "LLM app with
images". dMLX already has the hard parts: a planned-edit API (`/v1/workflows/route` with `plan: true`),
memory safety, and an OpenAI-compatible images API. Exposing them as tools makes dMLX the image backend
for other agents, a niche the GUI tools do not fill.

## To do
- [ ] An MCP server (stdio and local HTTP) with tools: `generate`, `edit` (by result id or path), `plan`
      (route a chat message, returns the agent's decision), `list_results`, `get_result`.
- [ ] Results returned as file paths and ids, with the prompt and seed; long jobs report progress.
- [ ] Same memory checks as the app; refusals come back as readable tool errors.
- [ ] Docs: using dMLX from Claude Code / Claude Desktop / LM Studio.

## Done when
A local agent (e.g. Claude Code) makes and edits an image through dMLX over MCP, end to end.
