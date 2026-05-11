#!/usr/bin/env bash
# sync-notes.sh — Sync markdown notes from topics/ to site/content/blog/
# Preserves subdirectory hierarchy. Generates _index.md for Hugo sections.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SITE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$SITE_DIR")"
TOPICS_DIR="$PROJECT_ROOT/topics"
CONTENT_DIR="$SITE_DIR/content/blog"

declare -A TOPIC_MAP=(
  ["01-diffusion-models"]="diffusion-models"
  ["02-robot-manipulation"]="robot-manipulation"
  ["03-autonomous-driving"]="autonomous-driving"
  ["04-3d-vision"]="3d-vision"
)

declare -A TOPIC_NAMES=(
  ["01-diffusion-models"]="扩散模型"
  ["02-robot-manipulation"]="机器人操控"
  ["03-autonomous-driving"]="自动驾驶"
  ["04-3d-vision"]="3D视觉"
)

declare -A SUBDIR_SLUGS=(
  ["理论基础"]="theory"
  ["UMI"]="umi"
  ["VLA"]="vla"
  ["RL"]="rl"
  ["world-models"]="world-models"
  ["diffusion-for-AD"]="diffusion-for-ad"
)

declare -A SUBDIR_DISPLAY=(
  ["theory"]="理论基础"
  ["umi"]="UMI 系列"
  ["vla"]="VLA 与基础模型"
  ["rl"]="强化学习"
  ["world-models"]="世界模型"
  ["diffusion-for-ad"]="扩散模型在自动驾驶中的应用"
)

SKIP_DIRS=("原始素材")

sync_count=0

generate_index() {
  local dir="$1"
  local title="$2"
  local description="$3"
  cat > "$dir/_index.md" <<EOF
---
title: "$title"
description: "$description"
---
EOF
}

should_skip_dir() {
  local dirname="$1"
  [[ "$dirname" == _* ]] && return 0
  for skip in "${SKIP_DIRS[@]}"; do
    [[ "$dirname" == "$skip" ]] && return 0
  done
  return 1
}

get_subdir_slug() {
  local dirname="$1"
  if [[ -n "${SUBDIR_SLUGS[$dirname]+x}" ]]; then
    echo "${SUBDIR_SLUGS[$dirname]}"
  else
    echo "$dirname" | tr '[:upper:]' '[:lower:]' | tr ' ' '-'
  fi
}

copy_with_frontmatter() {
  local src_file="$1"
  local dest_file="$2"
  local slug="$3"
  local display_name="$4"
  local subdir_name="${5:-}"

  if head -1 "$src_file" | grep -q "^---"; then
    cp "$src_file" "$dest_file"
  else
    local filename
    filename=$(basename "$src_file" .md)
    local tags="[\"$slug\""
    [[ -n "$subdir_name" ]] && tags="$tags, \"$subdir_name\""
    tags="$tags]"

    {
      echo "---"
      echo "title: \"$filename\""
      echo "date: $(date -r "$src_file" +%Y-%m-%d 2>/dev/null || date +%Y-%m-%d)"
      echo "tags: $tags"
      echo "summary: \"来自 ${display_name} 研究笔记\""
      echo "draft: false"
      echo "---"
      echo ""
      cat "$src_file"
    } > "$dest_file"
  fi
  ((sync_count++)) || true
}

for topic_prefix in "${!TOPIC_MAP[@]}"; do
  slug="${TOPIC_MAP[$topic_prefix]}"
  display_name="${TOPIC_NAMES[$topic_prefix]}"
  target_dir="$CONTENT_DIR/$slug"

  source_dir=$(find "$TOPICS_DIR" -maxdepth 1 -type d -name "${topic_prefix}*" | head -1)
  if [[ -z "$source_dir" ]]; then
    echo "Warning: No directory found for $topic_prefix"
    continue
  fi

  notes_dir="$source_dir/notes"
  if [[ ! -d "$notes_dir" ]]; then
    echo "Warning: No notes/ dir in $source_dir"
    continue
  fi

  # Clean and recreate topic directory
  rm -rf "$target_dir"
  mkdir -p "$target_dir"

  # Generate topic-level _index.md
  generate_index "$target_dir" "$display_name" "${display_name}相关研究笔记"

  # Process files directly in notes/ root (no subdirectory)
  while IFS= read -r src_file; do
    local_name=$(basename "$src_file")
    [[ "$local_name" == _* ]] && continue
    filename=$(basename "$src_file" .md)
    copy_with_frontmatter "$src_file" "$target_dir/${filename}.md" "$slug" "$display_name"
  done < <(find "$notes_dir" -maxdepth 1 -name "*.md" -type f)

  # Process subdirectories
  while IFS= read -r subdir; do
    subdir_name=$(basename "$subdir")

    if should_skip_dir "$subdir_name"; then
      echo "  Skipping: $subdir_name"
      continue
    fi

    subdir_slug=$(get_subdir_slug "$subdir_name")
    sub_target="$target_dir/$subdir_slug"
    mkdir -p "$sub_target"

    # Generate _index.md for subdirectory section
    local_display="${SUBDIR_DISPLAY[$subdir_slug]:-$subdir_name}"
    generate_index "$sub_target" "$local_display" "${display_name} — ${local_display}"

    # Sync all .md files from this subdirectory (recursive, skip _ prefixed)
    while IFS= read -r src_file; do
      [[ "$(basename "$src_file")" == _* ]] && continue
      filename=$(basename "$src_file" .md)
      copy_with_frontmatter "$src_file" "$sub_target/${filename}.md" \
        "$slug" "$display_name" "$local_display"
    done < <(find "$subdir" -name "*.md" -type f)
  done < <(find "$notes_dir" -maxdepth 1 -mindepth 1 -type d)
done

echo "Synced $sync_count notes to $CONTENT_DIR"
