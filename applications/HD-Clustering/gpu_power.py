import subprocess
import time
import threading
from collections import deque

BASELINE_PASSIVE_POWER_W = 23.5  # passive baseline in Watts

class GPUPowerMonitor:
    """Monitor GPU power consumption during application execution."""
    
    def __init__(self, gpu_id=0, sampling_interval=0.1):
        self.gpu_id = gpu_id
        self.sampling_interval = sampling_interval
        self.power_readings = deque()
        self.active_power_readings = deque()
        self.monitoring = False
        self.monitor_thread = None
        
    def _get_power_reading(self):
        """Get current GPU power draw in watts."""
        try:
            result = subprocess.run(
                ['nvidia-smi', f'--id={self.gpu_id}', 
                 '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
            pass
        return None
    
    def compute_active_power(self, power_w):
        return max(0.0, power_w - BASELINE_PASSIVE_POWER_W)
    
    def _monitor_loop(self):
        """Continuously record power readings."""
        while self.monitoring:
            power = self._get_power_reading()
            if power is not None:
                active_power = self.compute_active_power(power)
                self.active_power_readings.append(active_power)
                self.power_readings.append(power)
            time.sleep(self.sampling_interval)
    
    def start(self):
        """Start monitoring GPU power consumption."""
        if self.monitoring:
            return
        
        self.power_readings.clear()
        self.active_power_readings.clear()
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop(self):
        """Stop monitoring and return statistics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        
        if not self.power_readings:
            return None
        
        readings = list(self.power_readings)
        total_energy = sum(readings) * self.sampling_interval / 3600  # Convert to Wh

        readings = list(self.active_power_readings)
        total_energy = sum(readings) * self.sampling_interval / 3600  # Convert to Wh
        
        return {
            'total_energy_wh': total_energy,
            'total_energy_joules': total_energy * 3600,
            'avg_power_w': sum(readings) / len(readings),
            'max_power_w': max(readings),
            'min_power_w': min(readings),
            'num_samples': len(readings),
            'duration_seconds': len(readings) * self.sampling_interval
        }


