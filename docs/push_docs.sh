#!/bin/bash

set -ex

make docs

ref=$(git rev-parse HEAD)
originalBranch=$(git rev-parse --abbrev-ref HEAD)
# Function to switch back to the original branch
function cleanup {
    echo "Cleaning up worktree"
    git worktree remove --force /tmp/earth2grid-pages
}
git worktree add /tmp/earth2grid-pages pages
trap cleanup EXIT

rsync -av --delete --exclude ".git" docs/_build/html/ /tmp/earth2grid-pages/
touch /tmp/earth2grid-pages/.nojekyll

cd /tmp/earth2grid-pages
git add -A
git commit --no-verify -m "update doc from $ref"
echo "To update the website: git push origin pages"
