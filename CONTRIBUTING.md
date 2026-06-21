# Contributing to FEZrs

Thank you for helping improve FEZrs. Contributions should improve reliability, maintainability, test coverage, or scientific correctness while preserving the current project architecture.

## Development Workflow

1. Fork the project on GitHub.
2. Clone your fork locally.
3. Create a new branch for your work:

```bash
git checkout -b feature/something-new
```

4. Make focused changes that match the existing FEZrs structure and coding style.
5. Add or update tests in the `tests` folder for your changes.
6. Run the relevant tests before opening a pull request.
7. Push your branch to your fork.
8. Create a pull request on GitHub.

## Commit Rules

Use clear commit messages that start with an uppercase verb.

Examples:

```text
ADD spectral index validation
FIX raster reader error handling
UPDATE calculator tests
REMOVE unused helper
```

Keep commits focused. Avoid mixing unrelated features, bug fixes, and refactors in the same commit.

## Testing

Every new feature, bug fix, or behavior change should include tests in the `tests` folder.

Tests should:

- Cover the expected behavior.
- Cover failure paths when possible.
- Avoid network access.
- Avoid filesystem access unless the feature requires it.
- Use mocks or fixtures for external dependencies.

## AI-Oriented Contributions

AI-oriented development is welcome when it supports FEZrs goals, such as remote sensing workflows, scientific image processing, automation, validation, documentation, testing, or developer tooling.

For AI-generated or AI-assisted changes:

- Verify the scientific or technical behavior with tests.
- Check generated code for incorrect assumptions, unused dependencies, unsafe file handling, or hidden behavior changes.
- Keep the implementation readable and maintainable.
- Mention important AI-assisted design choices in the pull request when they affect behavior, calculations, or user-facing output.

## Pull Requests

Create a pull request with:

- A short summary of the change.
- A full description of what changed and why.
- Notes about tests that were added or updated.
- Any important behavior changes, limitations, or scientific assumptions.

Pull requests should be focused and easy to review. If a larger architectural change is needed, please discuss it before implementing it.

## Project Guidelines

- Preserve backward compatibility whenever possible.
- Do not reorganize the project structure without prior approval.
- Do not introduce new dependencies without explaining why they are needed.
- Prefer readable, explicit Python code.
- Use type hints for new public functions and methods.
- Preserve array shapes, data types, and numerical stability in scientific calculations.
- Do not modify `README.md` unless the change is explicitly requested.
