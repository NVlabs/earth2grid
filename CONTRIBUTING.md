# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

## Getting started

1. Install the pre-commit hooks: `pre-commit install`
1. Checkout a feature branch `git checkout -b <some feature>
1. When writing code, group changes logically into commits. Messy commits are
   usually a result of excessive multitasking. If you work on one thing at a
   time, then your commits will be cleaner.
    1. Name each commit "[imperative verb] [subject]" (e.g. "add earth2grid.some_modules"). Make sure it fits onto one line.
    1. Please provide context in the commit message. Start by explaining the
        previous state of the code and why this needed changing. Focus on motivating
        rather than explaining the changeset.
    1. run the test suite: `make unittest`
    1. run the docs: `make docs`
1. push the code to the repo `git push -u origin your-branch-name` and open an MR
1. ask for one reviewer.

## Tips


## Deploying

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in CHANGELOG.md).
Then run:

```
$ poetry run bump2version patch # possible: major / minor / patch
$ git push
$ git push --tags
```

GitHub Actions will then deploy to PyPI if tests pass.
