#!/bin/bash

originalBranch=$(git rev-parse --abbrev-ref HEAD)
# Function to switch back to the original branch
function switchBack {
    echo "Switching back to $originalBranch"
    git checkout $originalBranch
}
git checkout pages
trap switchBack EXIT
rm -r public/ && cp -r docs/_build/html public
git add -f public/
git commit -a -m "update doc from $ref"
git push origin pages
