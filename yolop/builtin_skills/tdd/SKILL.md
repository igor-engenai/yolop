---
name: tdd
description: Build behavior in small red-green-refactor cycles.
---
Use test-driven development.

1. Write one failing test for one observable behavior.
2. Confirm that the test fails for the expected reason.
3. Write the minimum implementation that makes it pass.
4. Refactor only while all tests pass.
5. Repeat with the next behavior.

Prefer tests through public interfaces. Do not test private implementation details.
