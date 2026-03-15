# [Song] 
# This script parses the output of compile.sh (passed as argv[1]; by default ./result.txt)
# and prints one TSV (tab-separated-value) row per trace log file with timing stats and command counts.
#
# compile.sh greps stats from each trace*.log file and prints them to stdout in order:
#   "Processing trace_X.log file..."   <- file delimiter line
#   memory_system_cycles <N>            <- one value, global simulation clock
#   idle_cycles <N>                     <- one line per channel (32 channels total)
#   precharged_cycles <N>               <- one line per channel
#   active_cycles <N>                   <- one line per channel
#   num_ACT_commands <N>                <- one line per channel
#   ...
#
# Output columns (tab-separated):
#   filename | total_time_ms | avg_active_ms | avg_precharged_ms | utilization_pct | cmd_counts...
#
# Timing conversion:
#   total_cycles / 2,000,000      -> ms  (GDDR6 @ 2 GHz)
#   active/precharged/idle / 32   -> per-channel average (32 channels in AiM simulator)
#   utilization = 100 - (avg_idle_cycles / total_cycles) * 100

import sys
file = open(sys.argv[1], 'r')
lines = file.readlines()
log = ""
total_cycles = -1
total_idle_cycles = -1
total_active_cycles = -1
total_precharged_cycles = -1
commands = ["ACT",
            "PREA",
            "PRE",
            "RD",
            "WR",
            "RDA",
            "WRA",
            "REFab",
            "REFpb",
            "ACT4",
            "ACT16",
            "PRE4",
            "MAC",
            "MAC16",
            "AF16",
            "EWMUL16",
            "RDCP",
            "WRCP",
            "WRGB",
            "RDMAC16",
            "RDAF16",
            "WRMAC16",
            "WRA16",
            "TMOD",
            "SYNC",
            "EOC"]
command_count = {}
for command in commands:
    command_count[command] = 0
for line in lines:
    # [Song]
    # "Processing <filename> file..." line from compile.sh output (results.txt) acts as a file delimiter.
    # When we see it, flush the accumulated stats of the PREVIOUS file (if any),
    # then reset all counters and record the new filename.
    # Stats for the last file are handled after the loop (lines below).
    if "Processing" in line:
        if total_cycles != -1:
            print(f"{log}\t{total_cycles/2000000.000}\t{total_active_cycles/32.00/2000000.000}\t{total_precharged_cycles/32.00/2000000.000}\t{100.00-(total_idle_cycles/32.00/total_cycles)*100.00}\t", end="")
            for command in commands:
                print(f"{command_count[command]}\t", end="")
                command_count[command] = 0
            print()
            total_cycles = -1
            total_idle_cycles = -1
            total_active_cycles = -1
            total_precharged_cycles = -1
        log = line.split()[1]
    # [Song] memory_system_cycles is a single global value (not per-channel), so just assign.
    if "memory_system_cycles" in line:
        total_cycles = int(line.split()[1])
    # idle/active/precharged_cycles appear once per channel (32 times total), so accumulate.
    # Sentinel -1 distinguishes "first channel" from a valid 0-cycle value.
    if "idle_cycles" in line:
        if total_idle_cycles == -1:
            total_idle_cycles = int(line.split()[1])
        else:
            total_idle_cycles += int(line.split()[1])
    if "active_cycles" in line:
        if total_active_cycles == -1:
            total_active_cycles = int(line.split()[1])
        else:
            total_active_cycles += int(line.split()[1])
    if "precharged_cycles" in line:
        if total_precharged_cycles == -1:
            total_precharged_cycles = int(line.split()[1])
        else:
            total_precharged_cycles += int(line.split()[1])
    for command in commands:
        if "num_" + command + "_commands" in line:
            command_count[command] += int(line.split()[1])
            
# [Song] Flush the last file's stats (no subsequent "Processing" line to trigger it).
if total_cycles != -1:
    print(f"{log}\t{total_cycles/2000000.000}\t{total_active_cycles/32.00/2000000.000}\t{total_precharged_cycles/32.00/2000000.000}\t{100.00-(total_idle_cycles/32.00/total_cycles)*100.00}\t", end="")
    for command in commands:
        print(f"{command_count[command]}\t", end="")
    print()