set -e  # Exit on error

# Parse command line arguments
ANSWERS_FILE="results/openend/answers_Qwen_Qwen3-32B_openend_longcontext_20260306_003434.jsonl"
TEST_FILE="${TEST_FILE:-dataset/test/open_end_qa_set.jsonl}"
JUDGE_CONFIG="${JUDGE_CONFIG:-configs/llm_judge.yaml}"
JUDGE_SERVER="${JUDGE_SERVER:-vllm}"
OUTPUT_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --answers-file)
            ANSWERS_FILE="$2"
            shift 2
            ;;
        --test-file)
            TEST_FILE="$2"
            shift 2
            ;;
        --judge-config)
            JUDGE_CONFIG="$2"
            shift 2
            ;;
        --judge-server)
            JUDGE_SERVER="$2"
            shift 2
            ;;
        --output-file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$ANSWERS_FILE" ]; then
    echo "Error: --answers-file is required"
    exit 1
fi

if [ -z "$TEST_FILE" ]; then
    echo "Error: --test-file is required"
    exit 1
fi

if [ -z "$JUDGE_CONFIG" ]; then
    echo "Error: --judge-config is required"
    exit 1
fi

# Validate files exist
if [ ! -f "$ANSWERS_FILE" ]; then
    echo "Error: Answers file not found: $ANSWERS_FILE"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "Error: Test file not found: $TEST_FILE"
    exit 1
fi

if [ ! -f "$JUDGE_CONFIG" ]; then
    echo "Error: Judge config not found: $JUDGE_CONFIG"
    exit 1
fi

# Build evaluation command
EVAL_ARGS=(
    --answers-file "$ANSWERS_FILE"
    --test-file "$TEST_FILE"
    --judge-config "$JUDGE_CONFIG"
    --judge-server "$JUDGE_SERVER"
)

if [ -n "$OUTPUT_FILE" ]; then
    EVAL_ARGS+=(--output-file "$OUTPUT_FILE")
fi

# Run evaluation
echo "Starting LLM-as-Judge evaluation..."
echo "  Answers file: $ANSWERS_FILE"
echo "  Test file: $TEST_FILE"
echo "  Judge config: $JUDGE_CONFIG"
echo "  Judge server: $JUDGE_SERVER"
if [ -n "$OUTPUT_FILE" ]; then
    echo "  Output file: $OUTPUT_FILE"
fi
echo ""

python -m src.evaluate "${EVAL_ARGS[@]}"

echo ""
echo "✅ Evaluation completed!"
