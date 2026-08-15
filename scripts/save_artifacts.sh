#!/usr/bin/env bash
# 保存实验产物到三个位置：
#   1. AutoDL 数据盘  — 通过 experiments -> /root/autodl-tmp/experiments 符号链接自动完成
#   2. GitHub         — 小文件复制到 results/ 目录后 git commit & push
#   3. HuggingFace Hub — 大文件（LoRA 权重、SFT 数据集、评测报告）上传到 HF repo
#
# 用法:
#   bash scripts/save_artifacts.sh results                   # 复制小文件到 results/
#   bash scripts/save_artifacts.sh hf                        # 上传大文件到 HuggingFace Hub
#   bash scripts/save_artifacts.sh all                       # 上述全部
#   bash scripts/save_artifacts.sh results --commit          # 复制 + git add + git commit
#   bash scripts/save_artifacts.sh results --commit --push   # 复制 + git add + commit + push
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
EXPERIMENTS_DIR="$PROJECT_ROOT/experiments"
RESULTS_DIR="$PROJECT_ROOT/results"

# HuggingFace repo ID（改成你自己的用户名）
HF_USER="${HF_USER:-your-hf-username}"
HF_DATASET_REPO="${HF_USER}/doproj-sft-data"
HF_MODEL_REPO="${HF_USER}/doproj-sft-lora"
HF_EVAL_REPO="${HF_USER}/doproj-eval-results"

DO_COMMIT=false
DO_PUSH=false
ACTION=""

while [[ $# -gt 0 ]]; do
    case $1 in
        results|hf|all) ACTION="$1"; shift ;;
        --commit)       DO_COMMIT=true; shift ;;
        --push)         DO_PUSH=true; DO_COMMIT=true; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [ -z "$ACTION" ]; then
    echo "用法: bash scripts/save_artifacts.sh {results|hf|all} [--commit] [--push]"
    exit 1
fi

# ---------------------------------------------------------------------------
# _copy_with_size_check: 复制文件到 results/，超过 50MB 则跳过并警告
# ---------------------------------------------------------------------------
_copy_with_size_check() {
    local src="$1" dest="$2" label="$3"
    local max_size=52428800  # 50 MB (GitHub 警告阈值)
    local size
    size=$(stat -c%s "$src" 2>/dev/null || stat -f%z "$src" 2>/dev/null || echo 0)
    if [ "$size" -ge "$max_size" ]; then
        echo "  跳过 $label (${size} bytes, >50MB, 不适合 git — 请用 HF 上传)"
        return
    fi
    cp "$src" "$dest"
    echo "  复制 $label (${size} bytes)"
}

# ---------------------------------------------------------------------------
# copy_results: 将 experiments/ 中的小文件复制到 results/ 供 git 追踪
# ---------------------------------------------------------------------------
copy_results() {
    echo "=========================================="
    echo "  复制小产物到 results/"
    echo "=========================================="

    if [ ! -d "$EXPERIMENTS_DIR" ]; then
        echo "[跳过] experiments/ 目录不存在"
        return
    fi

    mkdir -p "$RESULTS_DIR"

    for exp_dir in "$EXPERIMENTS_DIR"/*/; do
        [ -d "$exp_dir" ] || continue
        exp_name=$(basename "$exp_dir")
        dest_dir="$RESULTS_DIR/$exp_name"
        mkdir -p "$dest_dir"

        # 顶层小文件
        for f in config.yaml summary.md train_summary.json train_config.yaml \
                 collect_config.yaml split.json summary.json split_eval_report.json \
                 eval_report.json adapter_config.json; do
            if [ -f "$exp_dir/$f" ]; then
                cp "$exp_dir/$f" "$dest_dir/"
                echo "  复制 $exp_name/$f"
            fi
        done

        # metrics/ 目录下的小 JSON（per_task_results.json 只复制 <5MB 的）
        if [ -d "$exp_dir/metrics" ]; then
            mkdir -p "$dest_dir/metrics"
            for f in "$exp_dir"/metrics/*.json; do
                [ -f "$f" ] || continue
                fname=$(basename "$f")
                if [ "$fname" = "per_task_results.json" ]; then
                    _copy_with_size_check "$f" "$dest_dir/metrics/$fname" "$exp_name/metrics/$fname"
                else
                    cp "$f" "$dest_dir/metrics/"
                    echo "  复制 $exp_name/metrics/$fname"
                fi
            done
        fi

        # analysis/ 目录
        if [ -d "$exp_dir/analysis" ]; then
            mkdir -p "$dest_dir/analysis"
            cp -r "$exp_dir"/analysis/* "$dest_dir/analysis/" 2>/dev/null || true
            echo "  复制 $exp_name/analysis/"
        fi

        # SFT 数据采集产物（jsonl + meta.json，文件不大，追踪到 git）
        for f in train.jsonl holdout_train.jsonl; do
            if [ -f "$exp_dir/$f" ]; then
                _copy_with_size_check "$exp_dir/$f" "$dest_dir/$f" "$exp_name/$f"
            fi
        done
        for f in "$exp_dir"/task_*.jsonl "$exp_dir"/task_*.meta.json; do
            [ -f "$f" ] || continue
            fname=$(basename "$f")
            _copy_with_size_check "$f" "$dest_dir/$fname" "$exp_name/$fname"
        done
    done

    # 任务切分
    if [ -d "$PROJECT_ROOT/data/task_splits" ]; then
        mkdir -p "$RESULTS_DIR/task_splits"
        cp "$PROJECT_ROOT"/data/task_splits/*.json "$RESULTS_DIR/task_splits/" 2>/dev/null || true
        echo "  复制 data/task_splits/*.json"
    fi

    echo ""
    echo "复制完成。results/ 目录内容："
    find "$RESULTS_DIR" -type f | sort | sed "s|$PROJECT_ROOT/||"
    echo ""

    if [ "$DO_COMMIT" = true ]; then
        echo "=========================================="
        echo "  Git 提交"
        echo "=========================================="

        # AutoDL 新实例未配置 git 身份，设置默认值（仅限本仓库）
        if ! git config user.email >/dev/null 2>&1; then
            git config user.email "doproj@autodl.local"
            git config user.name "DoProj"
            echo "  已设置 git 身份: DoProj <doproj@autodl.local>"
        fi

        # 先推送之前未推送的提交（push 失败时本地会积压，下次运行先清掉）
        if [ "$DO_PUSH" = true ]; then
            UNPUSHED=$(git log --oneline origin/main..HEAD 2>/dev/null | wc -l)
            if [ "$UNPUSHED" -gt 0 ]; then
                echo "  发现 $UNPUSHED 个未推送提交，先尝试推送..."
                git push || {
                    echo "  [错误] 推送失败（GitHub 连接问题？），不再创建新提交避免重复。"
                    echo "  网络恢复后重跑本脚本即可自动推送。"
                    exit 1
                }
                echo "  已推送 $UNPUSHED 个积压提交。"
            fi
        fi

        git add results/ data/task_splits/ 2>/dev/null || git add results/
        if git diff --cached --quiet; then
            echo "没有新变更需要提交。"
        else
            git commit -m "save experiment results to results/"
            echo "已提交。"
            if [ "$DO_PUSH" = true ]; then
                echo "推送到远程..."
                git push || {
                    echo "  [错误] 推送失败，提交已保存在本地。"
                    echo "  网络恢复后重跑本脚本即可自动推送。"
                    exit 1
                }
                echo "已推送。"
            fi
        fi
    fi
}

# ---------------------------------------------------------------------------
# upload_hf: 上传大文件到 HuggingFace Hub
# 需要: export HF_TOKEN=xxx
# ---------------------------------------------------------------------------
upload_hf() {
    echo "=========================================="
    echo "  上传大产物到 HuggingFace Hub"
    echo "=========================================="

    if [ -z "${HF_TOKEN:-}" ]; then
        echo "[错误] 未设置 HF_TOKEN 环境变量"
        echo "  获取 token: https://huggingface.co/settings/tokens"
        echo "  设置: export HF_TOKEN=hf_xxxxxxxx"
        echo "跳过 HuggingFace 上传。"
        return 1
    fi

    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

    # --- 1. SFT 数据集 ---
    SFT_DIR="$EXPERIMENTS_DIR/sft_collect_airline"
    if [ -f "$SFT_DIR/train.jsonl" ]; then
        echo ""
        echo "[1/3] 上传 SFT 数据集到 $HF_DATASET_REPO ..."
        python3 -c "
from huggingface_hub import HfApi, create_repo
import os
api = HfApi(token=os.environ['HF_TOKEN'])
repo_id = '${HF_DATASET_REPO}'
try:
    create_repo(repo_id, repo_type='dataset', token=os.environ['HF_TOKEN'])
except Exception:
    pass
api.upload_file(
    path_or_fileobj='${SFT_DIR}/train.jsonl',
    path_in_repo='train.jsonl',
    repo_id=repo_id, repo_type='dataset')
print('  上传 train.jsonl')
for fname in ['split.json', 'summary.json', 'collect_config.yaml']:
    fpath = '${SFT_DIR}/' + fname
    if os.path.exists(fpath):
        api.upload_file(path_or_fileobj=fpath, path_in_repo=fname,
                        repo_id=repo_id, repo_type='dataset')
        print(f'  上传 {fname}')
print(f'SFT 数据集: https://huggingface.co/datasets/{repo_id}')
"
    else
        echo "[跳过] SFT 数据集不存在"
    fi

    # --- 2. LoRA adapter ---
    LORA_DIR="$EXPERIMENTS_DIR/sft_lora"
    if [ -f "$LORA_DIR/adapter_model.safetensors" ]; then
        echo ""
        echo "[2/3] 上传 LoRA adapter 到 $HF_MODEL_REPO ..."
        python3 -c "
from huggingface_hub import HfApi, create_repo
import os
api = HfApi(token=os.environ['HF_TOKEN'])
repo_id = '${HF_MODEL_REPO}'
try:
    create_repo(repo_id, repo_type='model', token=os.environ['HF_TOKEN'])
except Exception:
    pass
api.upload_folder(
    folder_path='${LORA_DIR}',
    repo_id=repo_id, repo_type='model',
    ignore_patterns=['checkpoint-*', '*.bin'])
print(f'LoRA adapter: https://huggingface.co/{repo_id}')
"
    else
        echo "[跳过] LoRA adapter 不存在"
    fi

    # --- 3. 评测报告 ---
    echo ""
    echo "[3/3] 上传评测报告到 $HF_EVAL_REPO ..."
    python3 -c "
from huggingface_hub import HfApi, create_repo
import os
api = HfApi(token=os.environ['HF_TOKEN'])
repo_id = '${HF_EVAL_REPO}'
try:
    create_repo(repo_id, repo_type='dataset', token=os.environ['HF_TOKEN'])
except Exception:
    pass
exp_base = '${EXPERIMENTS_DIR}'
for exp_name in sorted(os.listdir(exp_base)):
    exp_path = os.path.join(exp_base, exp_name)
    if not os.path.isdir(exp_path):
        continue
    for fname in ['eval_report.json', 'split_eval_report.json']:
        fpath = os.path.join(exp_path, fname)
        if os.path.exists(fpath):
            api.upload_file(path_or_fileobj=fpath,
                            path_in_repo=f'{exp_name}/{fname}',
                            repo_id=repo_id, repo_type='dataset')
            print(f'  上传 {exp_name}/{fname}')
print(f'评测报告: https://huggingface.co/datasets/{repo_id}')
" 2>/dev/null || echo "[跳过] 评测报告上传失败或不存在"

    echo ""
    echo "HuggingFace 上传完成。"
}

case $ACTION in
    results) copy_results ;;
    hf)      upload_hf ;;
    all)     copy_results; upload_hf ;;
esac
