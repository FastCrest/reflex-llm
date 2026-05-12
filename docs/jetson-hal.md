# Jetson Hardware Abstraction Layer

All hardware queries read directly from sysfs/procfs — no NVIDIA SDK dependency.

## Power Management

### Source: `src/jetson/power.cpp`

### Power Modes

| Mode | Watts | GPU Max MHz | Recommended use |
|------|-------|-------------|-----------------|
| MAXN (0) | 25W | 1300 | Maximum performance, active cooling required |
| 15W (1) | 15W | 900 | Balanced, small heatsink |
| 10W (2) | 10W | 600 | Low power, fanless possible |
| 7W (3) | 7W | 400 | Minimum, battery operation |

### Reading Power State

```cpp
PowerState ps = read_power_state();
// ps.mode:            POWER_MAXN / POWER_15W / POWER_10W / POWER_7W
// ps.watts:           25 / 15 / 10 / 7
// ps.gpu_freq_mhz:    current GPU frequency
// ps.emc_freq_mhz:    memory controller frequency
// ps.cpu_freq_mhz:    max CPU frequency
// ps.cpu_online:      number of online CPU cores
```

### sysfs Paths

| What | Path |
|------|------|
| GPU current frequency | `/sys/devices/17000000.ga10b/devfreq/17000000.ga10b/cur_freq` |
| GPU max frequency | `/sys/devices/17000000.ga10b/devfreq/17000000.ga10b/max_freq` |
| EMC (memory) frequency | `/sys/kernel/debug/bpmp/debug/clk/emc/rate` |
| CPU frequency | `/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq` |
| CPU online | `/sys/devices/system/cpu/cpuN/online` |
| Power mode | `nvpmodel -q` (popen) |

### Setting Power Mode

```cpp
set_power_mode(POWER_MAXN);  // calls: nvpmodel -m 0
lock_clocks();                // calls: jetson_clocks
```

## Thermal Management

### Source: `src/jetson/thermal.cpp`

### Reading Temperature

```cpp
ThermalState ts = read_thermal();
// ts.cpu_temp_c:   CPU temperature (°C)
// ts.gpu_temp_c:   GPU temperature (°C)
// ts.board_temp_c: Board temperature (°C)
// ts.throttling:   true if any zone > 85°C
```

Reads from `/sys/devices/virtual/thermal/thermal_zone*/temp` and matches zone type names ("CPU-therm", "GPU-therm", "Tboard_tegra").

### Adaptive Backoff

```cpp
int delay_us = thermal_backoff_us(ts);
```

| Temperature | Backoff | Effect |
|------------|---------|--------|
| < 80°C | 0 | Full speed |
| 80–85°C | 10 ms | Pre-throttle — slight slowdown |
| 85–90°C | 50 ms | Throttle zone — noticeable slowdown |
| 90–95°C | 100 ms | Critical — significant slowdown |
| > 95°C | 200 ms | Emergency — near shutdown |

Called every 10 tokens in the decode loop (not every token — sysfs reads are slow, ~100μs each).

## System Info

### Source: `src/jetson/sysinfo.cpp`

### One-Time Probe

```cpp
JetsonInfo info = probe_jetson();
print_jetson_info(info);
```

Output:
```
╔══════════════════════════════════════╗
║   Jetson LLM Runtime v0.1            ║
╠══════════════════════════════════════╣
║ L4T:    36.4       CUDA: 12.6       ║
║ SMs:    16          Cores: 1024      ║
║ RAM:    7633  MB    CMA: 768  MB    ║
║ NVMe:   42000 MB free               ║
╚══════════════════════════════════════╝
```

Reads:
- `/etc/nv_tegra_release` → L4T version
- `cudaRuntimeGetVersion()` → CUDA version
- `cudaGetDeviceProperties()` → SM count, compute capability
- `/proc/meminfo` → RAM, CMA
- `df` command → NVMe free space

### Live Stats

```cpp
LiveStats s = read_live_stats();
print_live_stats(s);
```

Output (single-line, carriage return for in-place update):
```
[RAM 3200/7633 MB | GPU 75% @ 1300 MHz | 52.3°C | 25.4 tok/s]
```

Reads:
- `/proc/meminfo` → RAM used/total
- `/sys/devices/17000000.ga10b/devfreq/17000000.ga10b/load` → GPU utilization %
- `/sys/devices/17000000.ga10b/devfreq/17000000.ga10b/cur_freq` → GPU MHz
- Thermal zones → GPU temperature
- `tokens_per_sec` set by engine
