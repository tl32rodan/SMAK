# Log Analyzer Design

`LogAnalyzer` reads structured log files and provides in-memory filtering and aggregation.

## Log Format

Each line must follow:

```
YYYY-MM-DD HH:MM:SS [LEVEL] message text
```

Example:

```
2024-01-15 10:23:01 [ERROR] connection refused: host unreachable
2024-01-15 10:24:30 [INFO] connection established
```

Lines that do not match this pattern are silently skipped during `parse()`.

## Operations

- `parse()` — reads the log file, returns a `list[LogEntry]`. Results are cached; subsequent
  calls to `count_by_level()` and `filter_by_level()` reuse the cache.
- `count_by_level()` — returns `{level: count}` frequency map. Level strings are
  case-preserved (e.g., `"ERROR"`, `"INFO"`).
- `filter_by_level(level)` — returns entries whose level matches `level`
  (comparison is case-insensitive).

## Data Model

```python
@dataclass
class LogEntry:
    timestamp: str   # e.g. "2024-01-15 10:23:01"
    level: str       # e.g. "ERROR"
    message: str     # e.g. "connection refused"
```

## Known Limitations

See `issues/log-analyzer-known-issues.md` for tracked bugs and design constraints.
