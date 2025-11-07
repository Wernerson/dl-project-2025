# DL 2025

> by Harald Semmelrock, Felix Schatzl, Sebastian Brunner


## How to...

### Add an external library / repo

1. `cd src/libs`
2. `git submodule add --name <name> -f <repo_url> ./<name>`
3. add `<name>` to `libs` in `cfg/config.yaml`
4. import library in code with `libs.<name>`