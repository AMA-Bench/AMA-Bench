import json
import re
import argparse
from pathlib import Path
from typing import List, Set, Dict, Any
from tqdm import tqdm


def parse_step_references(text: str) -> Set[int]:
    """
    Parse all step references from text.
    
    Matches patterns like:
    - "step 0", "step 1", "at step 4"
    - "between step 0 and step 2"
    - "from step 0 to step 11"
    - "step 0:", "step 1:", "step 2:"
    
    Args:
        text: Text to parse
        
    Returns:
        Set of step indices mentioned in the text
    """
    steps = set()
    
    if not text:
        return steps
    
    # Pattern 1: "step X" or "step X:" (where X is a number)
    # Matches: "step 0", "step 1", "at step 4", "step 0:", "step 1:"
    step_pattern = r'\bstep\s+(\d+)\b'
    for match in re.finditer(step_pattern, text, re.IGNORECASE):
        step_num = int(match.group(1))
        steps.add(step_num)
    
    # Pattern 2: "between step X and step Y" or "from step X to step Y"
    # Matches: "between step 0 and step 2", "from step 0 to step 11"
    range_pattern = r'(?:between|from)\s+step\s+(\d+)\s+(?:and|to)\s+step\s+(\d+)'
    for match in re.finditer(range_pattern, text, re.IGNORECASE):
        start_step = int(match.group(1))
        end_step = int(match.group(2))
        # Include all steps in the range (inclusive)
        steps.update(range(start_step, end_step + 1))
    
    return steps


def extract_relevant_turn_indices(qa_pair: Dict[str, Any]) -> List[int]:
    """
    Extract relevant turn indices from a QA pair by parsing question and answer.
    
    Args:
        qa_pair: QA pair dictionary with 'question' and 'answer' fields
        
    Returns:
        Sorted list of turn indices that contain the answer
    """
    question = qa_pair.get("question", "")
    answer = qa_pair.get("answer", "")
    
    # Parse steps from both question and answer
    steps_from_question = parse_step_references(question)
    steps_from_answer = parse_step_references(answer)
    
    # Combine and sort
    all_steps = sorted(steps_from_question | steps_from_answer)
    
    return all_steps


def process_json_file(
    json_file: Path,
    output_dir: Path = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Process a single JSON file and add relevant_turn_indices to QA pairs.
    
    Args:
        json_file: Path to input JSON file
        output_dir: Output directory (if None, overwrites input file)
        dry_run: If True, only report what would be changed without writing
        
    Returns:
        Dictionary with processing statistics
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    qa_pairs = data.get("qa_pairs", [])
    if not qa_pairs:
        return {
            "file": json_file.name,
            "episode_id": data.get("episode_id", "unknown"),
            "num_qa": 0,
            "num_annotated": 0,
            "num_already_annotated": 0,
        }
    
    num_annotated = 0
    num_already_annotated = 0
    
    for qa_pair in qa_pairs:
        # Skip if already has relevant_turn_indices
        if "relevant_turn_indices" in qa_pair:
            num_already_annotated += 1
            continue
        
        # Extract relevant turn indices
        relevant_indices = extract_relevant_turn_indices(qa_pair)
        
        # Add to QA pair
        qa_pair["relevant_turn_indices"] = relevant_indices
        num_annotated += 1
    
    # Write output
    if not dry_run:
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / json_file.name
        else:
            output_path = json_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    return {
        "file": json_file.name,
        "episode_id": data.get("episode_id", "unknown"),
        "num_qa": len(qa_pairs),
        "num_annotated": num_annotated,
        "num_already_annotated": num_already_annotated,
    }


def process_directory(
    input_dir: Path,
    output_dir: Path = None,
    file_pattern: str = "*.json",
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """
    Process all JSON files in a directory.
    
    Args:
        input_dir: Input directory containing JSON files
        output_dir: Output directory (if None, overwrites input files)
        file_pattern: File pattern to match (default: "*.json")
        dry_run: If True, only report what would be changed without writing
        
    Returns:
        List of processing statistics for each file
    """
    json_files = sorted(input_dir.glob(file_pattern))
    
    results = []
    
    for json_file in tqdm(json_files, desc="Processing files"):
        try:
            result = process_json_file(json_file, output_dir, dry_run)
            results.append(result)
        except Exception as e:
            print(f"\nError: Error processing {json_file.name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "file": json_file.name,
                "error": str(e),
            })
    
    return results


def print_summary(results: List[Dict[str, Any]]):
    """Print processing summary statistics."""
    if not results:
        return
    
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    
    total_files = len(results)
    total_qa = sum(r.get("num_qa", 0) for r in results)
    total_annotated = sum(r.get("num_annotated", 0) for r in results)
    total_already_annotated = sum(r.get("num_already_annotated", 0) for r in results)
    errors = sum(1 for r in results if "error" in r)
    
    print(f"Total files processed: {total_files}")
    if errors > 0:
        print(f"  Error: Errors: {errors}")
    print(f"Total QA pairs: {total_qa}")
    print(f"  Success: Newly annotated: {total_annotated}")
    print(f"  Already annotated:  Already annotated: {total_already_annotated}")
    
    if total_qa > 0:
        annotation_rate = (total_annotated / total_qa) * 100
        print(f"\nAnnotation rate: {annotation_rate:.1f}%")
    
    # Show distribution of relevant_turn_indices counts
    if total_annotated > 0:
        print(f"\n{'='*60}")
        print("Sample annotations (first 5 files with new annotations):")
        print(f"{'='*60}")
        shown = 0
        for r in results:
            if r.get("num_annotated", 0) > 0 and shown < 5:
                print(f"\nFile: {r['file']} (episode: {r.get('episode_id', 'unknown')})")
                print(f"   Annotated {r['num_annotated']} QA pairs")
                shown += 1
    
    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Add relevant_turn_indices to BabyAI QA pairs by parsing step references"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="babyai_out_batch",
        help="Input directory containing JSON files (default: babyai_out_batch)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (default: overwrite input files)",
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="*.json",
        help="File pattern to match (default: *.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode - report what would be changed without writing files",
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve() if args.output else None
    
    try:
        results = process_directory(
            input_dir,
            output_dir,
            args.file_pattern,
            args.dry_run,
        )
        print_summary(results)
        
        if not args.dry_run:
            if output_dir:
                print(f"Success: Annotated files written to: {output_dir.absolute()}")
            else:
                print(f"Success: Input files updated in place: {input_dir.absolute()}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
