#!/usr/bin/env bash
# Regenerate this extension from a Strycast checkout.
# 1. Check out <ref> of Strycast in a temporary worktree and apply
#    upstream-drift.patch (Gufo's Qwen changes after Strycast's base).
# 2. make_spike.py copies the Qwen code and rewrites private includes;
#    rename_ns.py moves it to namespace strycast.
# 3. Commit that as "generated" in <out> and apply ports/*.patch, the API
#    differences between Strycast's core and Gufo's public extension API.
# Usage: tools/generate.sh <strycast checkout> <out dir> [ref, default origin/staging]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
strycast=$(cd "$1" && pwd); out=$2; ref=${3:-origin/staging}
[ ! -e "$out" ] || { echo "$out exists" >&2; exit 1; }
tmp=$(mktemp -d)
trap 'git -C "$strycast" worktree remove --force "$tmp/src" >/dev/null 2>&1 || true; rm -rf "$tmp"' EXIT
git -C "$strycast" worktree add -q --detach "$tmp/src" "$ref"
git -C "$tmp/src" -c user.name=generate -c user.email=generate@localhost am -q -3 "$here/upstream-drift.patch"
python3 "$here/make_spike.py" "$tmp/src" HEAD "$out"
python3 "$here/rename_ns.py" "$out" "$here/core-namespaces.txt"
git -C "$out" init -q
git -C "$out" add -A
git -C "$out" commit -q -m "generated from Strycast $(git -C "$strycast" rev-parse --short=8 "$ref")"
git -C "$out" am -q -3 "$here"/ports/*.patch
git -C "$out" log --oneline
