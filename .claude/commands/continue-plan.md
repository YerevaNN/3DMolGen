---
description: Load project plan or specific feature plan
---

Read the project plan index at docs/plans/PLAN.md.

**Filtering rules:**
- By default, ONLY read the "Current Focus" and "Backlog" sections
- SKIP the "Completed" section unless the argument is "history"
- If argument is "history", read the full file including Completed

**If an argument is provided ($ARGUMENTS):**
- If argument is "history": read full PLAN.md including Completed section
- Otherwise: find and read the matching feature plan file in docs/plans/
- Match by partial name (e.g., "logit" matches files containing "logit" in the name)
- If multiple matches, list them and ask which to load

**After reading:**
1. Summarize current state (progress %, completed items, blockers if any)
2. List the next 2-3 actionable items from the plan
3. Ask what to work on this session

**If no matching plan is found:**
List available plan files in docs/plans/ and ask which to load.
