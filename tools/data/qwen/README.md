# Qwen authored smoke corpora

These UTF-8 JSONL documents were authored for repository evaluation plumbing.
They contain Chinese/English prose, code, arithmetic, structured text, other
languages and explicit chat turns. They contain no downloaded benchmark data.
Project Apache-2.0 licensing applies. They are small smoke corpora, not a
representative benchmark, validation of knowledge, or executable code/math
accuracy tests. Perplexity measures next-token prediction of their exact text.

- `calibration.jsonl`: 12 documents; observe ranges using the BF16 graph only.
- `heldout.jsonl`: 12 distinct documents; compare candidates without fitting
  their activation ranges on this data.
- `benchmark.jsonl`: three generation prompts, including previous chat turns.
  Repeated-run consistency and text output do not prove answer correctness.
- `calibration_context.jsonl`: the calibration records plus fixed snapshots of
  README prose (1024 tokens) and chat-session C++ source (512 tokens).
- `heldout_context.jsonl`: distinct architecture/detection documentation
  snapshots of 2048 and 1024 tokens. Each snapshot records its repository
  source; stored text is stable even if the source file later changes.

Records require a unique string `id`, optional `category`, and exactly one of
`text` or `messages`. Messages contain `role` and `content`; roles are system,
user or assistant. A system message may occur only first. Plain text is encoded
without implicit BOS/EOS tokens. Chat scoring includes all supplied message
and delimiter tokens. It is full-transcript loss, not assistant-only loss.
Benchmark messages must end in user and receive the repository non-thinking
assistant prefix. No text is truncated unless `--max-tokens` is explicitly set;
the manifest records original lengths and truncation. State resets between
JSONL records. Bring a larger domain-specific corpus for release decisions.
