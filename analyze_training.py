"""
Training Analysis Utility

Analyze and visualize training logs from temporal model training.

Usage:
    python analyze_training.py <path_to_training_log.csv>
    python analyze_training.py training/temporal_model_20251108_180000/training_log.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path
import sys


def load_training_log(log_path: str) -> pd.DataFrame:
    """Load training log CSV into pandas DataFrame."""
    df = pd.read_csv(log_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def plot_training_progress(log_path: str, save_path: str = None):
    """
    Create comprehensive training progress plots.
    
    Args:
        log_path: Path to training_log.csv
        save_path: Optional path to save plot image
    """
    df = load_training_log(log_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Temporal Model Training Progress', fontsize=16, fontweight='bold')
    
    # Plot 1: Fitness over generations
    ax1 = axes[0, 0]
    ax1.plot(df['generation'], df['mean_fitness'], label='Mean', linewidth=2, color='blue')
    ax1.plot(df['generation'], df['max_fitness'], label='Max', alpha=0.7, color='green')
    ax1.plot(df['generation'], df['best_fitness_overall'], label='Best Overall', 
             linestyle='--', linewidth=2, color='red')
    ax1.fill_between(df['generation'], 
                      df['mean_fitness'] - df['std_fitness'],
                      df['mean_fitness'] + df['std_fitness'],
                      alpha=0.2, label='±1 Std', color='blue')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.set_title('Fitness Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle=':', alpha=0.3)
    
    # Plot 2: Completion rate
    ax2 = axes[0, 1]
    ax2.plot(df['generation'], df['completion_rate'], color='green', linewidth=2)
    ax2.fill_between(df['generation'], 0, df['completion_rate'], alpha=0.3, color='green')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Completion Rate (%)')
    ax2.set_title('Route Completion Rate')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    
    # Plot 3: Progressive difficulty
    ax3 = axes[1, 0]
    ax3.plot(df['generation'], df['target_min_distance'], label='Min Distance', 
             linewidth=2, color='orange')
    ax3.plot(df['generation'], df['target_max_distance'], label='Max Distance', 
             linewidth=2, color='darkorange')
    ax3.fill_between(df['generation'], df['target_min_distance'], df['target_max_distance'],
                      alpha=0.3, color='orange')
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Route Distance (m)')
    ax3.set_title('Progressive Difficulty Curriculum')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Training efficiency
    ax4 = axes[1, 1]
    ax4.plot(df['generation'], df['generation_time_s'], color='purple', linewidth=2)
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Time (seconds)')
    ax4.set_title('Time per Generation')
    ax4.grid(True, alpha=0.3)
    
    # Add average time line
    avg_time = df['generation_time_s'].mean()
    ax4.axhline(y=avg_time, color='red', linestyle='--', alpha=0.5, 
                label=f'Average: {avg_time:.1f}s')
    ax4.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved: {save_path}")
    
    plt.show()


def print_training_summary(log_path: str):
    """Print summary statistics from training log."""
    df = load_training_log(log_path)
    
    print("=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total generations: {len(df)}")
    print(f"Best fitness: {df['best_fitness_overall'].iloc[-1]:.2f} (Gen {df['best_generation'].iloc[-1]})")
    print(f"Final mean fitness: {df['mean_fitness'].iloc[-1]:.2f}")
    print(f"Final max fitness: {df['max_fitness'].iloc[-1]:.2f}")
    print(f"Final completion rate: {df['completion_rate'].iloc[-1]:.1f}%")
    print(f"Total training time: {df['generation_time_s'].sum() / 3600:.2f} hours")
    print(f"Average time/generation: {df['generation_time_s'].mean():.1f}s")
    print("=" * 60)
    
    # Find improvement milestones
    print("\nMILESTONES:")
    try:
        first_positive = df[df['max_fitness'] > 0]['generation'].iloc[0]
        print(f"  First positive fitness: Gen {first_positive}")
    except:
        print(f"  First positive fitness: Not yet achieved")
    
    try:
        first_completion = df[df['completions'] > 0]['generation'].iloc[0]
        print(f"  First completion: Gen {first_completion}")
    except:
        print(f"  First completion: Not yet achieved")
    
    try:
        fifty_percent = df[df['completion_rate'] >= 50.0]['generation'].iloc[0]
        print(f"  50% completion rate: Gen {fifty_percent}")
    except:
        print(f"  50% completion rate: Not yet achieved")
    
    try:
        fitness_500 = df[df['max_fitness'] >= 500.0]['generation'].iloc[0]
        print(f"  First 500+ fitness: Gen {fitness_500}")
    except:
        print(f"  First 500+ fitness: Not yet achieved")
    
    print("=" * 60)
    
    # Performance trends
    print("\nPERFORMANCE TRENDS (Last 10 generations):")
    if len(df) >= 10:
        recent = df.tail(10)
        print(f"  Mean fitness improvement: {recent['mean_fitness'].iloc[-1] - recent['mean_fitness'].iloc[0]:+.2f}")
        print(f"  Completion rate improvement: {recent['completion_rate'].iloc[-1] - recent['completion_rate'].iloc[0]:+.1f}%")
        print(f"  Best fitness improvement: {recent['best_fitness_overall'].iloc[-1] - recent['best_fitness_overall'].iloc[0]:+.2f}")
    print("=" * 60)


def export_summary_stats(log_path: str, output_path: str = None):
    """Export summary statistics to JSON file."""
    df = load_training_log(log_path)
    
    stats = {
        'total_generations': len(df),
        'best_fitness': float(df['best_fitness_overall'].iloc[-1]),
        'best_generation': int(df['best_generation'].iloc[-1]),
        'final_mean_fitness': float(df['mean_fitness'].iloc[-1]),
        'final_completion_rate': float(df['completion_rate'].iloc[-1]),
        'total_time_hours': float(df['generation_time_s'].sum() / 3600),
        'avg_time_per_gen': float(df['generation_time_s'].mean()),
        'fitness_progression': {
            'first_10_mean': float(df.head(10)['mean_fitness'].mean()),
            'last_10_mean': float(df.tail(10)['mean_fitness'].mean()),
            'overall_improvement': float(df.tail(10)['mean_fitness'].mean() - df.head(10)['mean_fitness'].mean())
        }
    }
    
    if output_path is None:
        output_path = Path(log_path).parent / 'training_summary.json'
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Summary statistics exported to: {output_path}")
    return stats


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_training.py <path_to_training_log.csv>")
        print("Example: python analyze_training.py training/temporal_model_20251108_180000/training_log.csv")
        sys.exit(1)
    
    log_path = sys.argv[1]
    
    if not Path(log_path).exists():
        print(f"Error: Log file not found: {log_path}")
        sys.exit(1)
    
    # Print summary
    print_training_summary(log_path)
    
    # Export summary stats
    export_summary_stats(log_path)
    
    # Create plots
    save_path = Path(log_path).parent / 'training_progress.png'
    plot_training_progress(log_path, str(save_path))
