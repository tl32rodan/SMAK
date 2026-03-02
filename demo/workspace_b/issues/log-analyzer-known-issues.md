---
symbol: log-parse-error
---

# Log Analyzer Known Issues

- ISSUE-101: `parse()` silently skips malformed log lines with no warning or count returned
  to the caller. Silent data loss can hide misconfigured log formats.
- ISSUE-102: `count_by_level()` preserves the original case of level strings (e.g. `"Error"`
  vs `"ERROR"`), while `filter_by_level()` compares case-insensitively. Callers must
  normalise level keys before correlating results from both methods.
- ISSUE-103: The entire log file is read into memory during `parse()`. Very large log files
  (multi-GB) will exhaust available memory; streaming is not supported.
