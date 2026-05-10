#!/usr/bin/env bash
# sync-notes.sh — Sync markdown notes from topics/ to site/content/blog/
# Copies .md files, injects Hugo front matter if missing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SITE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$SITE_DIR")"
TOPICS_DIR="$PROJECT_ROOT/topics"
CONTENT_DIR="$SITE_DIR/content/blog"

# Mapping: topics folder prefix → content subfolder
declare -A TOPIC_MAP=(
  ["01-diffusion-models"]="diffusion-models"
  ["02-robot-manipulation"]="robot-manipulation"
  ["03-autonomous-driving"]="autonomous-driving"
  ["04-3d-vision"]="3d-vision"
)

# Mapping: topics folder prefix → display name
declare -A TOPIC_NAMES=(
  ["01-diffusion-models"]="扩散模型"
  ["02-robot-manipulation"]="机器人操控"
  ["03-autonomous-driving"]="自动驾驶"
  ["04-3d-vision"]="3D视觉"
)

sync_count=0

for topic_prefix in "${!TOPIC_MAP[@]}"; do
  slug="${TOPIC_MAP[$topic_prefix]}"
  display_name="${TOPIC_NAMES[$topic_prefix]}"
  target_dir="$CONTENT_DIR/$slug"
  mkdir -p "$target_dir"

  # Find the actual directory (may have Chinese suffix in parens)
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

  # Sync all .md files from notes/
  find "$notes_dir" -name "*.md" -type f | while read -r src_file; do
    # Get relative path from notes dir
    rel_path="${src_file#$notes_dir/}"
    filename=$(basename "$src_file" .md)

    # Skip files starting with _ (raw materials)
    if [[ "$rel_path" == _* ]]; then
      continue
    fi

    # Flatten: use filename only (avoid nested dirs in content)
    dest_file="$target_dir/${filename}.md"

    # Check if file has front matter already
    if head -1 "$src_file" | grep -q "^---"; then
      cp "$src_file" "$dest_file"
    else
      # Inject front matter
      {
        echo "---"
        echo "title: \"$filename\""
        echo "date: $(date -r "$src_file" +%Y-%m-%d 2>/dev/null || date +%Y-%m-%d)"
        echo "tags: [\"$slug\", \"$display_name\"]"
        echo "summary: \"来自 ${display_name} 研究笔记\""
        echo "draft: false"
        echo "---"
        echo ""
        cat "$src_file"
      } > "$dest_file"
    fi

    ((sync_count++)) || true
  done
done

echo "Synced $sync_count notes to $CONTENT_DIR"
