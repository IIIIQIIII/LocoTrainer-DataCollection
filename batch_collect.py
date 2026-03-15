#!/usr/bin/env python3
"""
Batch Data Collection Script for MS-SWIFT Analysis
Collects LocoTrainer trajectories for 500 queries on 8xH100 GPUs
"""

import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import argparse

# Import after environment is verified
def setup_paths():
    """Add src to path so we can import locotrainer"""
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root / "src"))

setup_paths()

from locotrainer.agent import Agent
from locotrainer.config import Config
from locotrainer.repo import ensure_ms_swift_repo


def collect_single_query(query_data: dict, config: Config, gpu_id: int) -> dict:
    """Collect trajectory for a single query on a specific GPU"""

    query_id = query_data["id"]
    query_text = query_data["query"]
    category = query_data.get("category", "unknown")

    print(f"[GPU {gpu_id}] Starting {query_id}: {query_text[:80]}...")

    # Set GPU for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Create output directory for this query
    output_base = Path(config.output_dir) / query_id
    output_base.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    try:
        # Run agent
        agent = Agent(config)
        conversation = agent.run(query_text)

        elapsed = time.time() - start_time

        # Save trajectory
        trajectory_file = output_base / "trajectory.json"
        with open(trajectory_file, "w", encoding="utf-8") as f:
            json.dump({
                "query_id": query_id,
                "query": query_text,
                "category": category,
                "subcategory": query_data.get("subcategory", ""),
                "conversation": conversation,
                "metadata": {
                    "elapsed_seconds": elapsed,
                    "turns": len([m for m in conversation if m["role"] == "assistant"]),
                    "gpu_id": gpu_id,
                    "timestamp": datetime.now().isoformat(),
                    "model": config.model,
                    "max_turns": config.max_turns,
                    "max_tokens": config.max_tokens
                }
            }, f, indent=2, ensure_ascii=False)

        # Save markdown output if exists
        output_md = output_base / "output.md"
        if output_md.exists():
            print(f"[GPU {gpu_id}] ✓ {query_id} completed in {elapsed:.1f}s ({len(conversation)} messages)")

        return {
            "query_id": query_id,
            "status": "success",
            "elapsed": elapsed,
            "turns": len([m for m in conversation if m["role"] == "assistant"]),
            "gpu_id": gpu_id
        }

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)
        print(f"[GPU {gpu_id}] ✗ {query_id} failed after {elapsed:.1f}s: {error_msg}")

        # Save error info
        error_file = output_base / "error.json"
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump({
                "query_id": query_id,
                "query": query_text,
                "error": error_msg,
                "elapsed": elapsed,
                "gpu_id": gpu_id,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)

        return {
            "query_id": query_id,
            "status": "error",
            "error": error_msg,
            "elapsed": elapsed,
            "gpu_id": gpu_id
        }


def load_queries(query_file: Path) -> list:
    """Load queries from JSON file"""
    with open(query_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support both formats: {"queries": [...]} and direct array
    if isinstance(data, dict) and "queries" in data:
        return data["queries"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Invalid query file format")


def main():
    parser = argparse.ArgumentParser(description="Batch collect MS-SWIFT analysis trajectories")
    parser.add_argument("--queries", type=str, default="data/msswift_queries_500.json",
                       help="Path to queries JSON file")
    parser.add_argument("--output", type=str, default="./trajectories",
                       help="Output directory for trajectories")
    parser.add_argument("--gpus", type=int, default=8,
                       help="Number of GPUs to use")
    parser.add_argument("--max-turns", type=int, default=40,
                       help="Max agent turns per query")
    parser.add_argument("--max-tokens", type=int, default=16384,
                       help="Max tokens per response (for 200k context)")
    parser.add_argument("--workers-per-gpu", type=int, default=1,
                       help="Number of parallel workers per GPU")
    parser.add_argument("--start-idx", type=int, default=0,
                       help="Start from query index (for resume)")
    parser.add_argument("--end-idx", type=int, default=None,
                       help="End at query index (for partial runs)")
    parser.add_argument("--ms-swift-path", type=str, default="/workspace/ms-swift",
                       help="Path to ms-swift codebase")

    args = parser.parse_args()

    # Load queries
    query_file = Path(args.queries)
    if not query_file.exists():
        print(f"Error: Query file not found: {query_file}")
        sys.exit(1)

    queries = load_queries(query_file)

    # Apply start/end index
    if args.end_idx:
        queries = queries[args.start_idx:args.end_idx]
    else:
        queries = queries[args.start_idx:]

    print(f"📊 Loaded {len(queries)} queries")
    print(f"🖥️  Using {args.gpus} GPUs with {args.workers_per_gpu} workers each")
    print(f"⚙️  Max turns: {args.max_turns}, Max tokens: {args.max_tokens}")

    # Ensure ms-swift repo exists
    ms_swift_path = Path(args.ms_swift_path)
    if not ms_swift_path.exists():
        print(f"⚠️  MS-SWIFT not found at {ms_swift_path}, attempting to clone...")
        ensure_ms_swift_repo(str(ms_swift_path.parent))

    # Create base config
    config = Config.from_env()
    config.max_turns = args.max_turns
    config.max_tokens = args.max_tokens
    config.codebase = args.ms_swift_path
    config.output_dir = args.output

    # Validate API key
    if not config.api_key or config.api_key == "local":
        print("⚠️  Warning: Using local API (api_key='local')")

    print(f"🚀 Starting batch collection to: {args.output}")
    print(f"🤖 Model: {config.model}")
    print(f"🌐 Base URL: {config.base_url}")

    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Total workers
    total_workers = args.gpus * args.workers_per_gpu

    # Distribute queries to GPUs
    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=total_workers) as executor:
        futures = {}

        for i, query_data in enumerate(queries):
            gpu_id = (i % args.gpus)  # Round-robin GPU assignment
            future = executor.submit(collect_single_query, query_data, config, gpu_id)
            futures[future] = query_data["id"]

        # Progress tracking
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1

            status_icon = "✓" if result["status"] == "success" else "✗"
            elapsed_total = time.time() - start_time
            avg_time = elapsed_total / completed
            eta = avg_time * (len(queries) - completed)

            print(f"[{completed}/{len(queries)}] {status_icon} {result['query_id']} "
                  f"(GPU {result['gpu_id']}) - {result['elapsed']:.1f}s | "
                  f"ETA: {eta/60:.1f}m")

    # Final statistics
    total_elapsed = time.time() - start_time
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = len(results) - success_count

    print("\n" + "="*80)
    print("📈 Batch Collection Complete!")
    print("="*80)
    print(f"Total queries: {len(queries)}")
    print(f"Successful: {success_count} ({success_count/len(queries)*100:.1f}%)")
    print(f"Failed: {error_count} ({error_count/len(queries)*100:.1f}%)")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"Average time per query: {total_elapsed/len(queries):.1f}s")
    print(f"Output directory: {args.output}")

    # Save summary
    summary_file = Path(args.output) / "collection_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_queries": len(queries),
            "successful": success_count,
            "failed": error_count,
            "total_elapsed_seconds": total_elapsed,
            "avg_seconds_per_query": total_elapsed / len(queries),
            "config": {
                "gpus": args.gpus,
                "workers_per_gpu": args.workers_per_gpu,
                "max_turns": args.max_turns,
                "max_tokens": args.max_tokens,
                "model": config.model
            },
            "results": results,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n📄 Summary saved to: {summary_file}")

    if error_count > 0:
        print(f"\n⚠️  {error_count} queries failed. Check error.json files in output directory.")


if __name__ == "__main__":
    main()
