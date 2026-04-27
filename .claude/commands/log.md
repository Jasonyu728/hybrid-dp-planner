Read `CHANGELOG.md` in the project root, then append a new entry based on changes made in this conversation:

```
## [vN] YYYY-MM-DD
- `file/path.py`: 改了什么
- `file/path.py`: 改了什么
```

Rules:
- Version number increments from the last entry.
- One bullet per modified file, one short sentence describing the change.
- No need to explain why — just what changed.
- If no changes were made this conversation, say so and do nothing.

After updating CHANGELOG.md, create a git commit to snapshot this version:
1. Stage only the files that were modified in this conversation (do NOT use `git add -A` or `git add .`)
2. Also stage `CHANGELOG.md`
3. Commit with message: `[vN] <one-line summary of changes>`
4. Report the commit hash so the user can reference it for future rollback.
