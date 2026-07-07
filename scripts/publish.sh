#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

message="${*:-notes: update vault}"

ignored_path() {
  case "$1" in
    .obsidian/*|public/*|.quartz/*|node_modules/*) return 0 ;;
  esac
  return 1
}

collect_paths() {
  paths=()
  while IFS= read -r -d '' entry; do
    status="${entry:0:2}"
    path="${entry:3}"
    if [[ "$status" == R* || "$status" == C* ]]; then
      IFS= read -r -d '' renamed_path || true
      path="$renamed_path"
    fi
    ignored_path "$path" && continue
    paths+=("$path")
  done < <(git status --porcelain -z --untracked-files=normal)
}

stage_collected_paths() {
  ((${#paths[@]} > 0)) && git add --all -- "${paths[@]}"
  git restore --staged -- .obsidian public .quartz node_modules 2>/dev/null || true
}

echo "Staging note/source changes..."
collect_paths
stage_collected_paths
if git diff --cached --quiet; then
  echo "No changes staged."
else
  git diff --cached --name-status
  git commit -m "$message"
fi

echo "Building (sanity check)..."
npm run build

echo "Pushing to origin/main..."
git push origin HEAD:main
echo "Done. GitHub Pages will rebuild via Actions."
