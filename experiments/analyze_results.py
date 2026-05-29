#!/usr/bin/env python3
"""
Analyze and compare results from the test suite.

Usage:
    python experiments/analyze_results.py results/n5_<timestamp>
"""

import os
import sys
import re
import json
from pathlib import Path
from collections import defaultdict
import argparse


def parse_log_file(log_path):
    """Extract key metrics from a training log file."""
    metrics = {
        'final_success_rate': None,
        'final_episode': None,
        'avg_episode_length': None,
        'total_episodes': 0,
        'successful_episodes': 0,
    }

    if not os.path.exists(log_path):
        return metrics

    with open(log_path, 'r') as f:
        lines = f.readlines()

    # Parse final statistics
    for line in reversed(lines[-50:]):  # Check last 50 lines
        if 'Success rate:' in line:
            match = re.search(r'Success rate: ([\d.]+)%.*\((\d+)/(\d+)\)', line)
            if match:
                metrics['final_success_rate'] = float(match.group(1))
                metrics['successful_episodes'] = int(match.group(2))
                metrics['total_episodes'] = int(match.group(3))

        if 'Average steps per episode:' in line:
            match = re.search(r'Average steps per episode: ([\d.]+)', line)
            if match:
                metrics['avg_episode_length'] = float(match.group(1))

    return metrics


def extract_config_from_dirname(dirname):
    """Extract configuration info from directory name."""
    config = {
        'test_id': dirname.split('_')[0] if '_' in dirname else dirname,
        'name': dirname,
        'reward_type': 'unknown',
        'reduction': 'none',
    }

    # Identify reward type
    if 'soft_matching' in dirname:
        config['reward_type'] = 'soft_matching'
    elif 'potential' in dirname:
        config['reward_type'] = 'potential'
    elif 'bounty' in dirname:
        config['reward_type'] = 'bounty'
    elif 'all_rewards' in dirname:
        config['reward_type'] = 'all_combined'

    # Identify dimension reduction
    if 'four_band' in dirname:
        if 'local' in dirname:
            if 'k3' in dirname:
                config['reduction'] = 'four_band+local_k3'
            elif 'k5' in dirname:
                config['reduction'] = 'four_band+local_k5'
            else:
                config['reduction'] = 'four_band+local'
        else:
            config['reduction'] = 'four_band'
    elif 'full_matrix' in dirname:
        config['reduction'] = 'full_matrix'

    return config


def analyze_results_directory(results_dir):
    """Analyze all test results in a directory."""
    results = []

    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Error: Directory {results_dir} does not exist")
        return results

    # Find all subdirectories (test results)
    for test_dir in sorted(results_path.iterdir()):
        if not test_dir.is_dir():
            continue

        log_file = test_dir / 'training_sb3.log'
        config = extract_config_from_dirname(test_dir.name)
        metrics = parse_log_file(log_file)

        # Check if test is still running
        is_running = log_file.exists() and metrics['final_success_rate'] is None

        results.append({
            'config': config,
            'metrics': metrics,
            'path': str(test_dir),
            'is_running': is_running,
        })

    return results


def print_summary_table(results, section=None):
    """Print a formatted table of results."""
    if section:
        results = [r for r in results if r['config']['test_id'].startswith(section)]

    if not results:
        print("No results found")
        return

    # Header
    print("\n" + "=" * 100)
    if section:
        print(f"SECTION {section} RESULTS")
    else:
        print("ALL RESULTS")
    print("=" * 100)
    print(f"{'Test':<8} {'Name':<35} {'Success %':<12} {'Avg Steps':<12} {'Episodes':<10}")
    print("-" * 100)

    # Sort by success rate (descending)
    sorted_results = sorted(
        results,
        key=lambda x: x['metrics']['final_success_rate'] or 0,
        reverse=True
    )

    # Rows
    for result in sorted_results:
        test_id = result['config']['test_id']
        name = result['config']['name'][:35]
        success = result['metrics']['final_success_rate']
        avg_steps = result['metrics']['avg_episode_length']
        total_eps = result['metrics']['total_episodes']

        success_str = f"{success:.2f}%" if success is not None else "N/A"
        steps_str = f"{avg_steps:.1f}" if avg_steps is not None else "N/A"
        eps_str = f"{total_eps}" if total_eps > 0 else "N/A"

        print(f"{test_id:<8} {name:<35} {success_str:<12} {steps_str:<12} {eps_str:<10}")

    print("=" * 100)


def print_section_summary(results, section_num):
    """Print summary for a specific section."""
    section_results = [r for r in results if r['config']['test_id'].startswith(section_num)]

    if not section_results:
        return

    # Check if any results are complete
    complete_results = [r for r in section_results if r['metrics']['final_success_rate'] is not None]

    if not complete_results:
        print(f"\n{'='*80}")
        print(f"SECTION {section_num} SUMMARY")
        print('='*80)
        print(f"\n⏳ Tests still running - no complete results yet")
        print(f"   Found {len(section_results)} test(s) in progress")
        print('='*80)
        return

    print(f"\n{'='*80}")
    print(f"SECTION {section_num} SUMMARY")
    print('='*80)

    # Find best result
    best = max(complete_results, key=lambda x: x['metrics']['final_success_rate'] or 0)

    print(f"\n🏆 Best Configuration:")
    print(f"   Test: {best['config']['test_id']}")
    print(f"   Name: {best['config']['name']}")
    print(f"   Success Rate: {best['metrics']['final_success_rate']:.2f}%")

    if best['metrics']['avg_episode_length'] is not None:
        print(f"   Avg Steps: {best['metrics']['avg_episode_length']:.1f}")
    else:
        print(f"   Avg Steps: N/A")

    # Section-specific analysis
    if section_num == '1':
        print(f"\n📊 Reward Comparison:")
        reward_results = defaultdict(list)
        for r in section_results:
            reward_results[r['config']['reward_type']].append(r)

        for reward_type, tests in sorted(reward_results.items()):
            avg_success = sum(t['metrics']['final_success_rate'] or 0 for t in tests) / len(tests)
            print(f"   {reward_type:20s}: {avg_success:6.2f}% avg success")

    elif section_num == '2':
        print(f"\n📊 Dimension Reduction Comparison:")
        reduction_results = defaultdict(list)
        for r in section_results:
            reduction_results[r['config']['reduction']].append(r)

        for reduction, tests in sorted(reduction_results.items()):
            avg_success = sum(t['metrics']['final_success_rate'] or 0 for t in tests) / len(tests)
            print(f"   {reduction:25s}: {avg_success:6.2f}% avg success")

    print('='*80)


def generate_comparison_report(results, output_file):
    """Generate a detailed comparison report."""
    with open(output_file, 'w') as f:
        f.write("# MSSA Test Suite Results Analysis\n\n")

        # Overall summary
        f.write("## Overall Summary\n\n")
        f.write(f"Total tests completed: {len(results)}\n\n")

        successful_tests = [r for r in results if r['metrics']['final_success_rate'] is not None]
        if successful_tests:
            avg_success = sum(r['metrics']['final_success_rate'] for r in successful_tests) / len(successful_tests)
            f.write(f"Average success rate: {avg_success:.2f}%\n\n")

        # Best configuration
        if successful_tests:
            best = max(successful_tests, key=lambda x: x['metrics']['final_success_rate'])
            f.write("## 🏆 Best Overall Configuration\n\n")
            f.write(f"- Test: {best['config']['test_id']}\n")
            f.write(f"- Name: {best['config']['name']}\n")
            f.write(f"- Success Rate: {best['metrics']['final_success_rate']:.2f}%\n")
            f.write(f"- Average Steps: {best['metrics']['avg_episode_length']:.1f}\n")
            f.write(f"- Path: {best['path']}\n\n")

        # Section summaries
        for section in ['1', '2', '3', '4']:
            section_results = [r for r in results if r['config']['test_id'].startswith(section)]
            if not section_results:
                continue

            f.write(f"## Section {section} Results\n\n")
            f.write("| Test | Name | Success % | Avg Steps | Episodes |\n")
            f.write("|------|------|-----------|-----------|----------|\n")

            for r in sorted(section_results, key=lambda x: x['metrics']['final_success_rate'] or 0, reverse=True):
                test_id = r['config']['test_id']
                name = r['config']['name']
                success = r['metrics']['final_success_rate'] or 0
                steps = r['metrics']['avg_episode_length'] or 0
                eps = r['metrics']['total_episodes']

                f.write(f"| {test_id} | {name} | {success:.2f}% | {steps:.1f} | {eps} |\n")

            f.write("\n")

    print(f"\n✅ Detailed report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze MSSA test suite results')
    parser.add_argument('results_dir', help='Path to results directory')
    parser.add_argument('--section', help='Filter by section (1, 2, 3, or 4)', default=None)
    parser.add_argument('--output', help='Output file for detailed report', default=None)

    args = parser.parse_args()

    # Analyze results
    print(f"\n📊 Analyzing results in: {args.results_dir}")
    results = analyze_results_directory(args.results_dir)

    if not results:
        print("❌ No results found")
        return

    # Check status
    running = [r for r in results if r.get('is_running', False)]
    complete = [r for r in results if not r.get('is_running', False) and r['metrics']['final_success_rate'] is not None]

    print(f"✅ Found {len(results)} test(s)")
    if running:
        print(f"   ⏳ {len(running)} test(s) still running")
    if complete:
        print(f"   ✓ {len(complete)} test(s) completed")
    if not complete and not running:
        print(f"   ⚠️  No tests have started yet")

    # Print summary table
    if args.section:
        print_summary_table(results, section=args.section)
        print_section_summary(results, args.section)
    else:
        # Print all sections
        for section in ['1', '2', '3', '4']:
            section_results = [r for r in results if r['config']['test_id'].startswith(section)]
            if section_results:
                print_summary_table(results, section=section)
                print_section_summary(results, section)

    # Generate detailed report
    if args.output:
        generate_comparison_report(results, args.output)
    else:
        output_file = os.path.join(args.results_dir, 'analysis_report.md')
        generate_comparison_report(results, output_file)


if __name__ == '__main__':
    main()
