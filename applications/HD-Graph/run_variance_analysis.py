import subprocess
import time
import statistics
from gpu_power import GPUPowerMonitor

root_dir_str = "/home/xrouth2/hdc/HDC-monorepo/applications/HD-Graph"
def run_application(app_path, iterations, gpu_id=0, sampling_interval=0.01):
    runtimes = []
    power_stats = []
    total_energy_stats = []

    for i in range(iterations):
        print(f"Run {i + 1}/{iterations}...")

        # Start GPU power monitoring
        monitor = GPUPowerMonitor(gpu_id=gpu_id, sampling_interval=sampling_interval)
        monitor.start()

        # Measure runtime
        start_time = time.time()
        subprocess.run([app_path],  cwd=f"{root_dir_str}",  check=False)
        end_time = time.time()

        # Stop GPU power monitoring
        stats = monitor.stop()

        time.sleep(5)
        # Record runtime and power stats
        runtime = end_time - start_time
        runtimes.append(runtime)
        if stats:
            power_stats.append(stats['avg_power_w'])
            # Record total energy in joules
            total_energy_joules = stats['total_energy_joules']
            total_energy_stats.append(total_energy_joules)

        print(f"Run {i + 1} completed: Runtime = {runtime:.4f}s, Avg Power = {stats['avg_power_w']:.4f}W, Total Energy = {total_energy_joules:.4f}J")

    return runtimes, power_stats, total_energy_stats

def main():
    app_path = root_dir_str + "/host-gpu-approx"
    iterations = 25
    gpu_id = 0
    sampling_interval = 0.01

    print(f"Running {app_path} {iterations} times...")
    runtimes, power_stats, total_energy_stats = run_application(app_path, iterations, gpu_id, sampling_interval)

    # Calculate and display statistics
    runtime_mean = statistics.mean(runtimes)
    runtime_stdev = statistics.stdev(runtimes) if len(runtimes) > 1 else 0
    power_mean = statistics.mean(power_stats)
    power_stdev = statistics.stdev(power_stats) if len(power_stats) > 1 else 0

    # Compute total energy statistics
    total_energy_mean = statistics.mean(total_energy_stats)
    total_energy_variance = statistics.variance(total_energy_stats) if len(total_energy_stats) > 1 else 0

    print("\n--- Variance Analysis ---")
    print(f"Runtime: Mean = {runtime_mean:.4f}s, Std Dev = {runtime_stdev:.4f}s")
    print(f"GPU Power: Mean = {power_mean:.4f}W, Std Dev = {power_stdev:.4f}W")
    print(f"Total Energy: Mean = {total_energy_mean:.4f}J, Variance = {total_energy_variance:.4f}J")

if __name__ == "__main__":
    main()
