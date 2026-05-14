# Jetson Hardware Abstraction Layer

All hardware queries read directly from sysfs/procfs — no NVIDIA SDK dependency.

## Power Management

### Source: `src/jetson/power.cpp`

### Power Modes

`nvpmodel` mode ids are board- and BSP-specific. On the validated L4T R36 Orin
Nano Super, `sudo nvpmodel -m 1` reports `NV Power Mode: 25W`. Do not assume
that id `1` means 15 W on every Jetson SKU.

The runtime parses the wattage string from `nvpmodel -q` and reports the actual
mode as `Power: 25W mode, GPU @ 918/918 MHz, 6 CPUs online` when clocks are
locked.

### Reading Power State

```cpp
PowerState ps = read_power_state();
// ps.mode:            POWER_MAXN / POWER_15W / POWER_10W / POWER_7W / POWER_UNKNOWN
// ps.watts:           parsed from nvpmodel -q, e.g. 25
// ps.gpu_freq_mhz:    current GPU frequency
// ps.gpu_freq_max_mhz:max GPU frequency from devfreq
// ps.emc_freq_mhz:    memory controller frequency
// ps.cpu_freq_mhz:    max CPU frequency
// ps.cpu_online:      number of online CPU cores
```

### sysfs Paths

| What | Path |
|------|------|
| GPU current frequency | probes R35/R36 variants under `17000000.ga10b`, `17000000.gpu`, and `gpu.0` |
| GPU max frequency | same devfreq path variants as current frequency |
| EMC (memory) frequency | `/sys/kernel/debug/bpmp/debug/clk/emc/rate` |
| CPU frequency | `/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq` |
| CPU online | `/sys/devices/system/cpu/online`, fallback to `cpuN/online` |
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
║ SMs:    8           Cores: 1024      ║
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
- devfreq path variants under `17000000.ga10b`, `17000000.gpu`, and `gpu.0`
  → GPU utilization and MHz
- Thermal zones → GPU temperature
- `tokens_per_sec` set by engine
