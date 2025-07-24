---
mode: agent
---
# Code Refactoring Task

## Objective
Refactor the provided code to improve readability and maintainability while preserving the exact same functionality and pipeline flow.

## Requirements
1. **Preserve Pipeline**: Maintain the same execution flow and logic
2. **Add Clear Dividers**: Insert section dividers/comments to improve navigation
3. **Clean Output**: Remove unnecessary print statements, debug logs, and console outputs
4. **Concise Implementation**: Simplify verbose code while keeping it readable
5. **Consistent Style**: Apply consistent naming conventions and formatting

## Refactoring Guidelines
- Add section headers like:
    ```
    # ========== INITIALIZATION ==========
    # ========== DATA PROCESSING ==========
    # ========== MAIN LOGIC ==========
    ```
- Remove:
    - Debug print statements
    - Commented-out code
    - Redundant comments
    - Unused imports/variables
- Simplify:
    - Long conditional chains
    - Repetitive code blocks
    - Overly complex expressions

## Success Criteria
- Code executes identically to the original
- Improved readability with clear section divisions
- Reduced code length without sacrificing clarity
- No debugging artifacts remain