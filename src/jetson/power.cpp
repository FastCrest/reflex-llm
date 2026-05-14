// power.cpp — Jetson power mode and clock control via sysfs

#include "jllm_jetson.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cctype>

namespace jllm {

static int read_sysfs_int(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    int val = 0;
    fscanf(f, "%d", &val);
    fclose(f);
    return val;
}

static int read_sysfs_mhz(const char* path, int divisor) {
    int val = read_sysfs_int(path);
    return val >= 0 ? val / divisor : -1;
}

static int parse_watts_from_line(const char* line) {
    for (const char* p = line; *p; ++p) {
        if (!isdigit((unsigned char)*p)) continue;

        char* end = nullptr;
        long value = strtol(p, &end, 10);
        if ((*end == 'W' || *end == 'w') && value > 0 && value < 1000) {
            return (int)value;
        }
        p = end ? end - 1 : p;
    }
    return -1;
}

static PowerMode mode_from_watts(int watts) {
    if (watts >= 25) return POWER_MAXN;
    if (watts >= 15) return POWER_15W;
    if (watts >= 10) return POWER_10W;
    if (watts >= 7)  return POWER_7W;
    return POWER_UNKNOWN;
}

static int read_online_cpu_count() {
    FILE* f = fopen("/sys/devices/system/cpu/online", "r");
    if (f) {
        char line[256] = {};
        int count = 0;
        if (fgets(line, sizeof(line), f)) {
            const char* p = line;
            while (*p) {
                while (*p && !isdigit((unsigned char)*p)) ++p;
                if (!*p) break;

                char* end = nullptr;
                long first = strtol(p, &end, 10);
                long last = first;
                if (*end == '-') {
                    p = end + 1;
                    last = strtol(p, &end, 10);
                }
                if (first >= 0 && last >= first) {
                    count += (int)(last - first + 1);
                }
                p = end;
            }
        }
        fclose(f);
        if (count > 0) return count;
    }

    // Fallback for kernels without the aggregate cpu/online file.
    int count = 0;
    for (int i = 0; i < 12; i++) {
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/online", i);
        int online = read_sysfs_int(path);
        if ((i == 0 && online < 0) || online == 1) count++;
    }
    return count;
}

PowerState read_power_state() {
    PowerState ps = {};
    ps.mode = POWER_UNKNOWN;
    ps.watts = -1;

    // GPU frequency
    ps.gpu_freq_mhz = read_sysfs_mhz(
        "/sys/devices/17000000.ga10b/devfreq/17000000.ga10b/cur_freq", 1000000);
    ps.gpu_freq_max_mhz = read_sysfs_mhz(
        "/sys/devices/17000000.ga10b/devfreq/17000000.ga10b/max_freq", 1000000);

    // EMC (memory controller) frequency
    ps.emc_freq_mhz = read_sysfs_mhz(
        "/sys/kernel/debug/bpmp/debug/clk/emc/rate", 1000000);

    // CPU frequency (first online core)
    ps.cpu_freq_mhz = read_sysfs_mhz(
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", 1000);

    // Count online CPUs
    ps.cpu_online = read_online_cpu_count();

    // Power mode from nvpmodel
    FILE* f = popen("nvpmodel -q 2>/dev/null", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strstr(line, "MAXN")) {
                ps.mode = POWER_MAXN;
                ps.watts = 25;
                break;
            }

            int watts = parse_watts_from_line(line);
            if (watts > 0) {
                ps.watts = watts;
                ps.mode = mode_from_watts(watts);
                break;
            }
        }
        pclose(f);
    }

    return ps;
}

void set_power_mode(PowerMode mode) {
    char cmd[64];
    snprintf(cmd, sizeof(cmd), "nvpmodel -m %d 2>/dev/null", (int)mode);
    system(cmd);
}

void lock_clocks() {
    system("jetson_clocks 2>/dev/null");
}

}  // namespace jllm
