"""
Performance profiling utilities for satellite flight software.
Tracks cycle time and peak memory usage per stage.
"""

import time
import psutil
import os
from contextlib import contextmanager

class PerformanceProfiler:
    """Track cycle time and memory for flight software stages."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.stage_times = {}
        self.stage_memory = {}
        self.total_start_time = None
        self.total_start_memory = None
    
    def start_total(self):
        """Start tracking total mission time."""
        self.total_start_time = time.perf_counter()
        self.total_start_memory = self.process.memory_info().rss / 1024  # KB
    
    @contextmanager
    def profile_stage(self, stage_name):
        """Context manager for profiling a single stage."""
        # Record start
        start_time = time.perf_counter()
        start_memory = self.process.memory_info().rss / 1024  # KB
        
        try:
            yield
        finally:
            # Record end
            end_time = time.perf_counter()
            end_memory = self.process.memory_info().rss / 1024  # KB
            
            # Calculate metrics
            cycle_time_ms = (end_time - start_time) * 1000
            peak_memory_kb = end_memory
            
            # Store results
            if stage_name not in self.stage_times:
                self.stage_times[stage_name] = []
                self.stage_memory[stage_name] = []
            
            self.stage_times[stage_name].append(cycle_time_ms)
            self.stage_memory[stage_name].append(peak_memory_kb)
    
    def print_report(self):
        """Print performance metrics for all stages."""
        print("\n" + "=" * 70)
        print("PERFORMANCE METRICS")
        print("=" * 70)
        
        # Calculate aggregates
        total_time = 0
        peak_memory = 0
        
        print(f"\n{'Stage':<35} {'Avg Time (ms)':<15} {'Peak Memory (KB)':<15}")
        print("-" * 70)
        
        for stage in sorted(self.stage_times.keys()):
            times = self.stage_times[stage]
            memories = self.stage_memory[stage]
            
            avg_time = sum(times) / len(times)
            max_memory = max(memories)
            
            total_time += sum(times)
            peak_memory = max(peak_memory, max_memory)
            
            print(f"{stage:<35} {avg_time:>13.2f}   {max_memory:>13.1f}")
        
        print("-" * 70)
        print(f"{'TOTAL MISSION TIME':<35} {total_time:>13.2f}   {peak_memory:>13.1f}")
        print("=" * 70)
    
    def reset(self):
        """Reset all metrics."""
        self.stage_times.clear()
        self.stage_memory.clear()

def measure_model_size(model_path):
    """Get model file size in human-readable format."""
    from pathlib import Path
    size_bytes = Path(model_path).stat().st_size
    
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
