#!/usr/bin/env bash
# Download the community-fixed Qwen3 chat template
# Source: https://huggingface.co/eemin/Qwen-Fixed-Chat-Templates
# Fixes: empty-think poisoning, agentic tool call loops, Qwen3.8 regressions
# Version: v22.1 (2026-08-16) — confirmed for Qwen3.8-27B

set -euo pipefail

MODEL_DIR="${1:-$HOME/models/qwen3.8-27b-int4}"

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "❌ Model directory not found: $MODEL_DIR"
    echo "Usage: $0 /path/to/model/dir"
    exit 1
fi

echo "📥 Downloading fixed chat template for Qwen3.8..."
echo "   Source: eemin/Qwen-Fixed-Chat-Templates (v22.1, Apache-2.0)"
echo "   Target: $MODEL_DIR/chat_template.jinja"

# Backup existing template if present
if [[ -f "$MODEL_DIR/chat_template.jinja" ]]; then
    cp "$MODEL_DIR/chat_template.jinja" "$MODEL_DIR/chat_template.jinja.bak"
    echo "   Backed up existing template to chat_template.jinja.bak"
fi

# Download the fixed template
python3 -c "
from huggingface_hub import hf_hub_download
import shutil

path = hf_hub_download(
    repo_id='eemin/Qwen-Fixed-Chat-Templates',
    filename='chat_template.jinja',
    local_dir='/tmp/qwen_fixed_template'
)
shutil.copy(path, '$MODEL_DIR/chat_template.jinja')
print('✅ Fixed chat template downloaded')
"

echo ""
echo "✅ Done. Restart serve_model.py to apply."
echo ""
echo "What this fixes (eemin/Qwen-Fixed-Chat-Templates v22.1):"
echo "  - Empty Think Poisoning: model loops <tool_call><web_search> infinitely"
echo "  - Qwen3.8 empty think regression: blank <think></think> in history"
echo "  - Tool call format: restored native XML <function=name><parameter=k>v"
echo "  - Thinking bypass hallucination: </antThinking> appearing in output"
echo "  - enable_thinking=false crash fixed (official Qwen3.8 throws exception)"
echo "  - Two-tier agentic error escalation to break retry loops"
