# Contributing to DocuQuery

Thank you for your interest in contributing to DocuQuery. This project is designed as a production-oriented RAG system, and we welcome contributions that improve its performance, reliability, and usability.

---
## How to Contribute

### 1. Fork the Repository

Create your own fork of the repository and clone it locally.

---
### 2. Create a Feature Branch

Use a clear and descriptive branch name:
```text
feature/
fix/
improvement/
```
Examples:
```text
- feature/add-hybrid-search
- fix/retrieval-threshold-bug
- improvement/api-response-format
```

---
### 3. Make Changes

- Keep changes **focused and minimal**
- Follow existing project structure and naming conventions
- Add comments where necessary for clarity

---
### 4. Write Tests

All contributions should include appropriate test coverage:

- Unit tests for core logic
- Integration tests for pipelines or APIs
- Ensure all existing tests continue to pass

---
### 5. Run Tests

Validation Checklist:

Before submitting your changes, please ensure the following checks pass:

```bash
pytest
python -m app.run_pipeline
python -m eval.run_eval
uvicorn app.main:app --reload
```

---
### 6. Submit a Pull Request

When opening a PR, include:

- Clear description of the change
- Motivation for the change
- Any relevant screenshots or logs (if applicable)

---
## Pull Request Guidelines

- Keep PRs small and focused
- Avoid unrelated changes in a single PR
- Ensure code is clean and readable
- Respond to review comments promptly

All pull requests will be reviewed before merging.

---
## Code Style

- Follow consistent Python formatting
- Use meaningful variable and function names
- Keep functions modular and reusable
- Avoid unnecessary complexity

---
## Areas for Contribution

We actively welcome improvements in:

- Retrieval quality (ranking, filtering)
- Evaluation metrics and datasets
- Performance optimizations
- API improvements
- Documentation clarity
- Testing coverage

---
## Reporting Issues

If you encounter a bug or have a feature request:

- Use the GitHub Issues tab
- Provide clear steps to reproduce (for bugs)
- Include logs or error messages if available

---
## Contribution Philosophy

DocuQuery is built with a focus on:

- Reliability over shortcuts  
- Clarity over cleverness  
- Measurable improvements over assumptions  

Contributions aligning with these principles are highly encouraged.

---
## Author

Aditya Bisoyi  
Software Engineer focused on scalable systems and applied machine learning
