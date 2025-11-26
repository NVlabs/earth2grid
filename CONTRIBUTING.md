# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

## Developer Certificate of Origin

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

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

## Building the docs


```
uv sync --no-install-project --group doc --extra dev
source .venv/bin/activate
python3 setup.py build_ext --inplace
export PYTHONPATH=$PWD
make docs
```

To push
```
make push_docs
git push origin pages
```

## Releasing

1. Modify VERSION variable in setup.py
1. Update changelog
1. git commit -m "release VERSION"
1. git tag vVERSION
1. git push
1. git push --tags
