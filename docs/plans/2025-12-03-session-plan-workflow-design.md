# Feature: Session Plan Workflow

## Goal
Implement a persistent plan system with a top-level index and feature-specific plans, enabling Claude to maintain project state across sessions.

## Status
- Progress: 0%
- Started: 2025-12-03
- Blockers: None

## Design

### File Structure
```
docs/plans/
├── PLAN.md                           # Top-level index (lightweight)
├── 2025-12-01-constrained-conformer-smoke.md    # (existing)
├── 2025-12-01-constrained-conformer-debug-log.md # (existing)
├── 2025-12-02-v1-baseline-fixes.md              # (existing)
└── <future feature plans...>

.claude/
├── commands/
│   └── continue-plan.md              # Slash command
└── settings.local.json               # (existing)
```

### Top-level PLAN.md Structure
- **Current Focus**: Active feature plans with links and progress %
- **Backlog**: Future work items
- **Completed**: Finished features (filtered out by default to reduce context)

### Slash Command Behavior
- `/continue-plan` → Reads index (skips Completed section), shows current focus
- `/continue-plan <name>` → Loads specific feature plan by partial name match
- `/continue-plan history` → Includes Completed section

### Session Protocol (CLAUDE.md)
- At session start: Use `/continue-plan` to load context
- Before session end: Update active feature plan and index
- When starting new work: Create new feature plan with template

### Feature Plan Template
```markdown
# Feature: <Name>

## Goal
<1-2 sentence description>

## Status
- Progress: X%
- Started: YYYY-MM-DD
- Blockers: None | <list>

## Checklist
- [ ] Task 1
- [ ] Task 2
- [ ] Task 3

## Next Actions
1. <immediate next step>
2. <following step>

## Notes
<session notes, decisions, learnings>
```

## Checklist
- [ ] Create `.claude/commands/continue-plan.md`
- [ ] Create `docs/plans/PLAN.md` index with existing plans
- [ ] Append session protocol to `CLAUDE.md`
- [ ] Test `/continue-plan` command
- [ ] Test `/continue-plan <feature>` with partial match

## Next Actions
1. Create the slash command file
2. Create the PLAN.md index
3. Update CLAUDE.md with session protocol

## Notes
- Chose index-style for top-level to keep it lightweight
- Selective reading (skip Completed) to avoid context pollution
- Argument-based command for flexibility
