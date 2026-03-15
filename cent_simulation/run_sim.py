import os
import math
import pandas as pd
import argparse
import subprocess
import concurrent.futures
from cxl_latency import llama_latency, gpt_latency, vector_latency
from cent_power_calculator import DRAM_POWER, ACCEL_CYCLE, ACCEL_POWER, SRAM_POWER, CTRL_POWER, commands, isrs, power_calculator, command_processor, KILO, MEGA, GIGA, FREQ, WORD_SIZE, tRC, tBL, tCCDL, RV_COUNT, SB_RD_CYCLE, SB_WR_CYCLE, EXP_LANE_CYCLE, RV_RMSNorm_CYCLE, RV_ROTEmbed_CYCLE, RV_SFT_CYCLE_PIPELINE, RV_SFT_CYCLE_SINGLE
from utils import InOut_latency, n_heads, gqa_factor, embedding_size, ffn_size, TransformerBlock_number, minimal_channel_per_block, pipeline_parallel_mode_list, model_parallel_mode_list

def get_args():
    parser = argparse.ArgumentParser('run_scripts.py')
    parser.add_argument("--num_channels", type=int, help="Number of channels per device", default=32)
    parser.add_argument("--num_devices", type=int, help="Number of CXL devices", default=32)
    parser.add_argument("--PCIE_lanes", type=int, help="Number of PCIE lanes", default=144)
    parser.add_argument("--reuse_size", type=int, help="GB reuse size, depending on register number", default=32)
    parser.add_argument("--generate_trace_max_workers", type=int, help="maximum concurrent threads to generate traces, limited by memory", default=20)
    parser.add_argument("--run_simulation_max_workers", type=int, help="maximum concurrent threads to generate traces, limited by memory", default=4)
    parser.add_argument("--model", choices=["Llama2-7B", "Llama2-13B", "Llama2-70B"], help="LLM Model", required=True)
    parser.add_argument("--generate_trace", action="store_true", help="Generate traces")
    parser.add_argument("--simulate_trace", action="store_true", help="Simulate traces")
    parser.add_argument("--process_results", action="store_true", help="Process results")
    parser.add_argument("--update_csv", action="store_true", help="Update results to csv file")
    parser.add_argument("--simulation_result_path", type=str, help="Path to the result file", default="simulation_results.csv")
    parser.add_argument("--process_throughputs", action="store_true", help="average throughputs for various seqlen")
    parser.add_argument("--processed_result_path", type=str, help="Path to the final result file", default="processed_results.csv")
    parser.add_argument("--phase", choices=["end2end", "prefill", "decoding"], help="Phase of the model", default="end2end")
    parser.add_argument("--prefill", type=int, help="Prefill length", default=512)
    parser.add_argument("--decoding", type=int, help="Decoding length", default=3584)
    parser.add_argument("--seqlen", type=int, nargs='+', help="Sequence list")
    parser.add_argument("--seqlen_gap", type=int, help="Gap between sequence lengths", default=128)
    parser.add_argument("--model_parallel", action="store_true", help="Apply model parallelism")
    parser.add_argument("--inter-device-attention", action="store_true")
    args = parser.parse_args()
    return args


def factorize(n):
    # [Song] returns all divisors of n in sorted order
    # used to enumerate all valid TP x PP combinations. (TP side = # FC devices)
    # ex) num_devices=32 -> [1, 2, 4, 8, 16, 32] -> (TP=1,PP=32), (TP=2,PP=16), ..., (TP=32,PP=1)
    factors = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            factors.append(i)
            if i != n // i:
                factors.append(n // i)
    return sorted(factors)

def generate_trace(args, seqlen_list):

    print(f"Generating traces for {args.model} with {args.generate_trace_max_workers} threads...")

    if args.model == "GPT3-175B":
        model = "--GPT3-175B"
    elif args.model == "Llama2-70B" or "Llama3" in args.model:
        model = "--Llama-GQA"
    elif "Llama2" in args.model:
        model = "--Llama"

    commands_generate_traces = []
    # ex) Llama2-70B: (80-1)//32+1 = 3
    blocks_per_device = (TransformerBlock_number[args.model] - 1) // args.num_devices + 1
    # ex) 32channels / 3blocks = 10 channels/block
    channels_per_block = args.num_channels // blocks_per_device
    FC_devices_list = factorize(args.num_devices) # [Song] TP only.

    # [Song] Side Note: difference between TP(FC_devices=1) trace and PP trace
    # TP(FC_devices=1): --model-parallel flag, 1 block uses all 32 channels (FC_devices=1, so no split)
    # PP:               --pipeline-parallel flag, 32 channels split across blocks_per_device blocks via channels_per_block
    #                   each device handles multiple blocks sequentially, channels are divided among them


    # [Song] output embedding trace generation (hidden_dim x vocab_size GEMV)
    # generated once (not per seqlen) since embedding size is fixed regardless of KV cache size
    # TP: one trace per FC_devices value; embedding uses all channels across FC_devices devices (same as transformer block mapping)
    #     only FC_devices is needed here (not num_devices)
    # PP: one trace using channels_per_block channels (same allocation unit as transformer blocks on that device)
    seqlen = args.prefill + args.decoding
    if args.model_parallel:
        for FC_devices in FC_devices_list:
            if not os.path.exists(f"../trace/{args.num_channels}_channels_per_device/model_parallel_embedding/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt"):
                commands_generate_traces.append(["python3", "function_sim.py", model, "--n_heads", str(n_heads[args.model]), "--ffn_dim", str(ffn_size[args.model]), "--embedding", "--only-trace", "--num-channels", str(args.num_channels), "--FC-devices", str(FC_devices), "--model-parallel", "--seqlen", str(seqlen), "--op-trace", "--GEMV", "reuse-GB", "--reuse-size", str(args.reuse_size), "--trace-file", f"../trace/{args.num_channels}_channels_per_device/model_parallel_embedding/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt"])
    else:
        if not os.path.exists(f"../trace/{args.num_channels}_channels_per_device/pipeline_parallel_embedding/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt"):
            commands_generate_traces.append(["python3", "function_sim.py", model, "--n_heads", str(n_heads[args.model]), "--ffn_dim", str(ffn_size[args.model]), "--embedding", "--only-trace", "--num-channels", str(args.num_channels), "--channels-per-block", str(channels_per_block), "--pipeline-parallel", "--multi-tb-per-device", "--seqlen", str(seqlen), "--op-trace", "--GEMV", "reuse-GB", "--reuse-size", str(args.reuse_size), "--trace-file", f"../trace/{args.num_channels}_channels_per_device/pipeline_parallel_embedding/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt"])

    for seqlen in seqlen_list:
        if args.model_parallel:          
            for FC_devices in FC_devices_list:
                if not os.path.exists(f"../trace/{args.num_channels}_channels_per_device/model_parallel/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt"):
                    commands_generate_traces.append(["python3", "function_sim.py", model, "--n_heads", str(n_heads[args.model]), "--ffn_dim", str(ffn_size[args.model]), "--only-trace", "--num-channels", str(args.num_channels), "--FC-devices", str(FC_devices), "--model-parallel", "--seqlen", str(seqlen), "--op-trace", "--GEMV", "reuse-GB", "--reuse-size", str(args.reuse_size), "--trace-file", f"../trace/{args.num_channels}_channels_per_device/model_parallel/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt"])
                    if args.inter_device_attention:
                        commands_generate_traces[-1].append("--inter-device-attention")
                if not os.path.exists(f"../trace/{args.num_channels}_channels_per_device/model_parallel_FC/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt"):
                    commands_generate_traces.append(["python3", "function_sim.py", model, "--n_heads", str(n_heads[args.model]), "--ffn_dim", str(ffn_size[args.model]), "--only-FC", "--only-trace", "--num-channels", str(args.num_channels), "--FC-devices", str(FC_devices), "--model-parallel", "--seqlen", str(seqlen), "--op-trace", "--GEMV", "reuse-GB", "--reuse-size", str(args.reuse_size), "--trace-file", f"../trace/{args.num_channels}_channels_per_device/model_parallel_FC/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt"])
        else:
            if channels_per_block < minimal_channel_per_block[args.model]:
                raise ValueError(f"Channels per block {channels_per_block} is less than minimal channel per block {minimal_channel_per_block[args.model]}")
            if not os.path.exists(f"../trace/{args.num_channels}_channels_per_device/pipeline_parallel/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt"):
                commands_generate_traces.append(["python3", "function_sim.py", model, "--n_heads", str(n_heads[args.model]), "--ffn_dim", str(ffn_size[args.model]), "--only-trace", "--num-channels", str(args.num_channels), "--channels-per-block", str(channels_per_block), "--pipeline-parallel", "--multi-tb-per-device", "--seqlen", str(seqlen), "--op-trace", "--GEMV", "reuse-GB", "--reuse-size", str(args.reuse_size), "--trace-file", f"../trace/{args.num_channels}_channels_per_device/pipeline_parallel/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt"])

    # [Song] run all queued trace generation commands in parallel using a thread pool
    # max_workers controlled by --generate_trace_max_workers (passed as $threads in simulation.sh)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.generate_trace_max_workers) as executor:
        futures = [executor.submit(subprocess.run, cmd) for cmd in commands_generate_traces]
        for future in concurrent.futures.as_completed(futures):
            future.result()

def run_command(command, log_file):
    # [Song]
    # Run a single ramulator2 command and save its output to a .log file.
    # Lines starting with '[' are filtered out (maybe ramulator2 progress bar noise).
    print(command)
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    filtered_output = "\n".join(line for line in result.stdout.splitlines() if not line.startswith('['))
    with open(log_file, "w") as log:
        log.write(filtered_output)

def detect_emtpy_file(file):
    return os.stat(file).st_size == 0

def simulate_trace(args, seqlen_list):
    commands_simulate_traces = []

	# ../aim_simulator/build/ramulator2 -f ../aim_simulator/test/example.yaml -t ../trace/32_channels_per_device/pipeline_parallel/Llama2-7B/trace_8_channels_per_block_seqlen_1.txt 2>&1 | grep '^[^\[]' &> ../trace/32_channels_per_device/pipeline_parallel/Llama2-7B/trace_8_channels_per_block_seqlen_1.txt.log

    blocks_per_device = (TransformerBlock_number[args.model] - 1) // args.num_devices + 1
    channels_per_block = args.num_channels // blocks_per_device
    FC_devices_list = factorize(args.num_devices)

    # Embedding
    seqlen = args.prefill + args.decoding
    if args.model_parallel:
        for FC_devices in FC_devices_list:
            log_file = f"../trace/{args.num_channels}_channels_per_device/model_parallel_embedding/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt.log"
            if not os.path.exists(log_file) or detect_emtpy_file(log_file):
                trace_file = f"../trace/{args.num_channels}_channels_per_device/model_parallel_embedding/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt"
                command = f"../aim_simulator/build/ramulator2 -f ../aim_simulator/test/example.yaml -t {trace_file}"
                commands_simulate_traces.append((command, log_file))
    else:
        log_file = f"../trace/{args.num_channels}_channels_per_device/pipeline_parallel_embedding/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt.log"
        if not os.path.exists(log_file) or detect_emtpy_file(log_file):
            trace_file = f"../trace/{args.num_channels}_channels_per_device/pipeline_parallel_embedding/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt"
            command = f"../aim_simulator/build/ramulator2 -f ../aim_simulator/test/example.yaml -t {trace_file}"
            commands_simulate_traces.append((command, log_file))

    for seqlen in seqlen_list:
        if args.model_parallel:
            for FC_devices in FC_devices_list:
                for mode in ["model_parallel", "model_parallel_FC"]:
                    log_file = f"../trace/{args.num_channels}_channels_per_device/{mode}/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt.log"
                    if not os.path.exists(f"../trace/{args.num_channels}_channels_per_device/{mode}/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt.log") or detect_emtpy_file(f"../trace/{args.num_channels}_channels_per_device/{mode}/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt.log"):
                        trace_file = f"../trace/{args.num_channels}_channels_per_device/{mode}/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt"
                        command = f"../aim_simulator/build/ramulator2 -f ../aim_simulator/test/example.yaml -t {trace_file}"
                        commands_simulate_traces.append((command, log_file))
        else:
            for mode in ["pipeline_parallel"]:
                log_file = f"../trace/{args.num_channels}_channels_per_device/{mode}/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt.log"
                if not os.path.exists(log_file) or detect_emtpy_file(log_file):
                    trace_file = f"../trace/{args.num_channels}_channels_per_device/{mode}/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt"
                    command = f"../aim_simulator/build/ramulator2 -f ../aim_simulator/test/example.yaml -t {trace_file}"
                    commands_simulate_traces.append((command, log_file))
    
    # [Song]
    # Run all ramulator2 commands in parallel using a thread pool.
    # max_workers = --run_simulation_max_workers (passed as $1 threads in simulation.sh).
    # executor.submit(run_command, cmd, log) schedules run_command(cmd, log) on a worker thread.
    # as_completed() yields futures as they finish; future.result() re-raises any exception from the thread.
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.run_simulation_max_workers) as executor:
        futures = [executor.submit(run_command, cmd, log) for cmd, log in commands_simulate_traces]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def process_results(args):
    print("Processing results...")
    mode_list = model_parallel_mode_list if args.model_parallel else pipeline_parallel_mode_list
    for mode in mode_list:
        compile_dir = f"../trace/{args.num_channels}_channels_per_device/{mode}/{args.model}/"
        subprocess.run(["cp", "../trace/compile.sh", compile_dir])
        subprocess.run(["cp", "../trace/compile.py", compile_dir])

        # Run compile.sh and write output to result.txt
        result = subprocess.run(["bash", "compile.sh"], cwd=compile_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        with open(f"{compile_dir}/result.txt", "w") as result_file:
            result_file.write(result.stdout)
            result_file.write(result.stderr)

        # Run compile.py and write output to compiled_results.txt
        result = subprocess.run(["python3", "compile.py", "./result.txt"], cwd=compile_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        with open(f"{compile_dir}/compiled_results.txt", "w") as compiled_results_file:
            compiled_results_file.write(result.stdout)
            compiled_results_file.write(result.stderr)

def calculate_acc_latency(args, seqlen):
    latency = {}
    GQA_factor = 1.00 + 1.00 / gqa_factor[args.model]
    latency["RMSNorm_latency"] =  embedding_size[args.model] / 16.00 / 16.00 / args.num_channels * ACCEL_CYCLE["VEC"]    # EMB /16.00 /16.00 ADD
    latency["RMSNorm_latency"] += SB_RD_CYCLE + SB_WR_CYCLE + 1.00                              # 1 RED
    latency["RMSNorm_latency"] += RV_RMSNorm_CYCLE                                              # 1 RISCV
    latency["RMSNorm_latency"] = float(2.00 * latency["RMSNorm_latency"]) / float(FREQ / KILO)
    latency["Softmax_latency"] =  seqlen * n_heads[args.model] / 16.00 / args.num_channels * ACCEL_CYCLE["EXP"]        # TOK*HEAD /16.00 EXP, maybe args.num_channels = 32 > # of exponent processors and accumulators in pnm unit?
    latency["Softmax_latency"] += seqlen * n_heads[args.model] / 16.00 / args.num_channels * ACCEL_CYCLE["VEC"]        # TOK*HEAD /16.00 ADD
    latency["Softmax_latency"] += n_heads[args.model] * 1.00 * SB_RD_CYCLE                                     # HEAD RED
    latency["Softmax_latency"] += n_heads[args.model] * RV_SFT_CYCLE_PIPELINE                                  # HEAD RISCV
    latency["Softmax_latency"] = float(latency["Softmax_latency"]) / float(FREQ / KILO)
    latency["RotEmbed_latency"] = embedding_size[args.model] * RV_ROTEmbed_CYCLE                                 # EMB RISCV
    latency["RotEmbed_latency"] = float(GQA_factor * latency["RotEmbed_latency"]) / float(FREQ / KILO)
    return latency

def load_data_point(args, seqlen, FC_devices, channels_per_block, PCIe_lanes_per_device, blocks_per_device, embedding_latency, utilized_devices, pp, tp):

    if args.model_parallel:
        path = f"../trace/{args.num_channels}_channels_per_device/model_parallel/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt.log"
    else:
        path = f"../trace/{args.num_channels}_channels_per_device/pipeline_parallel/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt.log"
    stats = command_processor(path)
    pim_latency = stats["latency"]

    acc_latency_dict = calculate_acc_latency(args, seqlen)
    acc_latency = (acc_latency_dict["RMSNorm_latency"] + acc_latency_dict["Softmax_latency"] + acc_latency_dict["RotEmbed_latency"])
    transformer_block_latency = pim_latency + acc_latency

    if args.model_parallel:
        if "Llama" in args.model:
            block_cxl_latency, p2p_cxl_latency = llama_latency([embedding_size[args.model], ffn_size[args.model]], PCIe_lanes_per_device, FC_devices, args.num_devices)
            transformer_block_latency += block_cxl_latency
        else:
            raise NotImplementedError("gpt latency not fixed yet")
            # cxl_latency = gpt_latency([embedding_size[args.model], ffn_size[args.model]], PCIe_lanes_per_device, FC_devices, args.num_devices)
        embedding_latency_data = embedding_latency['model_parallel'][FC_devices]
        p2p_hops = pp - 1 
    else:
        p2p_cxl_latency = vector_latency(embedding_size[args.model], PCIe_lanes_per_device)
        p2p_hops = utilized_devices - 1
        embedding_latency_data = embedding_latency['pipeline_parallel'][channels_per_block]


    # [Song] FIXME: unit inconsistency in PP mode.
    # pim_latency is per-block but acc_latency seems to be per-device.
    # transformer_block_latency is per-block which adds pim_latency and acc_latency.
    #acc_latency = (acc_latency_dict["RMSNorm_latency"] + acc_latency_dict["Softmax_latency"] + acc_latency_dict["RotEmbed_latency"]) * blocks_per_device
    #transformer_block_latency = pim_latency + cxl_latency + acc_latency
    #token_latency = transformer_block_latency * TransformerBlock_number[args.model] + embedding_latency_data + InOut_latency

    token_latency = transformer_block_latency * TransformerBlock_number[args.model] + embedding_latency_data + InOut_latency + p2p_hops * p2p_cxl_latency
    throughput = 1000 / token_latency * pp

    energy_token = {}
    power_alldv = {}
    PCIE = embedding_size[args.model] * 10 + ffn_size[args.model] * 2 if args.model_parallel else embedding_size[args.model]
    energy_main, latency_main = power_calculator(stats, PCIE, n_heads[args.model], embedding_size[args.model], seqlen, gqa_factor[args.model])
    if args.model_parallel:
        pipeline_stages = args.num_devices // FC_devices
        FC_path = f"../trace/{args.num_channels}_channels_per_device/model_parallel_FC/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt.log"
        stats_FC = command_processor(FC_path)
        energy_FC, latency_FC = power_calculator(stats_FC, PCIE, n_heads[args.model], embedding_size[args.model], seqlen, gqa_factor[args.model])
        for comp in energy_main.keys():
            energy_token[comp] = (energy_main[comp] + energy_FC[comp] * (FC_devices - 1)) * TransformerBlock_number[args.model]
            power_alldv[comp] = (energy_main[comp] + energy_FC[comp] * (FC_devices - 1)) * pipeline_stages / stats["latency"]
    else:
        for comp in energy_main.keys():
            energy_token[comp] = energy_main[comp] * utilized_devices
            power_alldv[comp] = energy_main[comp] * utilized_devices / stats["latency"]
    total_energy = 0
    for comp in energy_token.keys():
        total_energy += energy_token[comp]
    total_power = 0
    for comp in power_alldv.keys():
        total_power += power_alldv[comp]
    device_utilization = 1.0 * utilized_devices / args.num_devices
                        
    new_result = {
        'Model': args.model,
        'Device number': args.num_devices,
        'Pipeline parallelism': pp,
        'Tensor parallelism': tp,
        'Channels per device': args.num_channels,
        'Channels per block': channels_per_block,
        'Sequence length': seqlen,
        'PIM latency': pim_latency,
        'CXL latency': p2p_cxl_latency,
        'Acc latency': acc_latency,
        'TransformerBlock latency': transformer_block_latency,
        'Embedding latency': embedding_latency_data,
        'Token latency (ms)': token_latency,
        'Throughput (tokens/s)': throughput,
        'Token energy (mJ)': total_energy,
        'Total power (W)': total_power,
        'Device utilization': device_utilization
    }
    new_result_df = pd.DataFrame([new_result])
    return new_result_df


def update_csv(args, seqlen_list):

    print("Updating simulation results to CSV file...")

    if os.path.exists(args.simulation_result_path):
        results_df = pd.read_csv(args.simulation_result_path)
    else:
        columns = ['Model', 'Device number', 'Pipeline parallelism', 'Tensor parallelism', 'Channels per device', 'Channels per block', 'Sequence length', 'PIM latency', 'CXL latency', 'Acc latency', 'TransformerBlock latency', 'Embedding latency', 'Token latency (ms)', 'Throughput (tokens/s)', 'Token energy (mJ)', 'Total power (W)', 'Device utilization']
        results_df = pd.DataFrame(columns=columns)

    embedding_latency = {'pipeline_parallel': {}, 'model_parallel': {}}

    if args.model_parallel:
        FC_devices_list = factorize(args.num_devices)

        # [Song] - FIXME
        # outer loop over FC_devices_list is redundant;
        # compiled_results.txt already contains one entry per FC_devices config.
        for FC_devices in FC_devices_list:
            embedding_compile_dir = f"../trace/{args.num_channels}_channels_per_device/model_parallel_embedding/{args.model}/"
            with open(f"{embedding_compile_dir}/compiled_results.txt", "r") as compiled_results_file:
                lines = compiled_results_file.readlines()
                for line in lines:
                    filename, latency = line.split()[0], line.split()[1]
                    FC_devices = int(filename.split('_')[1])
                    embedding_latency["model_parallel"][FC_devices] = float(latency)
    else:
        embedding_compile_dir = f"../trace/{args.num_channels}_channels_per_device/pipeline_parallel_embedding/{args.model}/"
        with open(f"{embedding_compile_dir}/compiled_results.txt", "r") as compiled_results_file:
            lines = compiled_results_file.readlines()
            for line in lines:
                filename, latency = line.split()[0], line.split()[1]
                channels_per_block = int(filename.split('_')[1])
                embedding_latency["pipeline_parallel"][channels_per_block] = float(latency)

    for seqlen in seqlen_list:

        PCIe_lanes_per_device = args.PCIE_lanes // args.num_devices

        if args.model_parallel:
            blocks_per_device = 1
            channels_per_block = args.num_channels // blocks_per_device
            utilized_devices = args.num_devices
            for FC_devices in FC_devices_list:
                pp = args.num_devices // FC_devices
                tp = FC_devices
                new_result_df = load_data_point(args, seqlen, FC_devices, channels_per_block, PCIe_lanes_per_device, blocks_per_device, embedding_latency, utilized_devices, pp, tp)
                results_df = pd.concat([results_df, new_result_df], ignore_index=True)
        else:
            pp = TransformerBlock_number[args.model]
            tp = 1
            pp_per_device = (pp - 1) // args.num_devices + 1
            blocks_per_device =  pp_per_device * (TransformerBlock_number[args.model] // pp)
            channels_per_block = args.num_channels // blocks_per_device
            if channels_per_block < minimal_channel_per_block[args.model]:
                continue
            utilized_devices = (TransformerBlock_number[args.model] - 1) // blocks_per_device + 1
            new_result_df = load_data_point(args, seqlen, 0, channels_per_block, PCIe_lanes_per_device, blocks_per_device, embedding_latency, utilized_devices, pp, tp)
            results_df = pd.concat([results_df, new_result_df], ignore_index=True)

    # Save the DataFrame to a CSV file
    results_df = results_df.drop_duplicates(subset=['Model', 'Device number', 'Pipeline parallelism', 'Tensor parallelism', 'Channels per device', 'Channels per block', 'Sequence length'])
    results_df = results_df.sort_values(by=['Model', 'Device number', 'Pipeline parallelism', 'Tensor parallelism', 'Channels per device', 'Channels per block', 'Sequence length'])
    results_df.to_csv(args.simulation_result_path, index=False)
    # print(results_df)

def process_throughputs(args):

    print("Processing results to CSV file...")

    if os.path.exists(args.simulation_result_path):
        df_simulation = pd.read_csv(args.simulation_result_path)
    else:
        raise ValueError(f"File {args.simulation_result_path} does not exist. Generate simulation results first.")
    
    if os.path.exists(args.processed_result_path):
        results_df = pd.read_csv(args.processed_result_path)
    else:
        columns = ['Model', 'Device number', 'Seqlen', 'Pipeline parallelism', 'Tensor parallelism', 'Phase', 'Total Latency (s)', 'Throughput (tokens/s)', 'Energy per Token (mJ)', 'Total power (W)']
        results_df = pd.DataFrame(columns=columns)


    if args.model_parallel:

        FC_devices_list = factorize(args.num_devices)

        for FC_Devices in FC_devices_list:

            pp = args.num_devices // FC_Devices
            tp = FC_Devices
            df = df_simulation[(df_simulation['Model'] == args.model) & (df_simulation['Pipeline parallelism'] == pp) & (df_simulation['Tensor parallelism'] == tp)]
            # print("tp", tp, len(df))

            if args.phase == "prefill":
                df = df[(df['Sequence length'] <= args.prefill)]
                seqlen = args.prefill
            elif args.phase == "decoding":
                df = df[((args.prefill + args.decoding) >= df['Sequence length']) & (df['Sequence length'] > args.prefill)]
                seqlen = args.decoding
            elif args.phase == "end2end":
                df = df[((args.prefill + args.decoding) >= df['Sequence length'])]
                seqlen = args.prefill + args.decoding

            # FIXME - This seems wrong..
            # average_throughput = df['Throughput (tokens/s)'].mean()
            average_energy = df['Token energy (mJ)'].mean()
            total_latency = df['Token latency (ms)'].mean() * seqlen / 1000
            average_throughput = (seqlen / total_latency) * pp
            total_power = df['Total power (W)'].mean()

            new_result = {
                'Model': args.model,
                'Device number': args.num_devices,
                'Seqlen': args.prefill + args.decoding,
                'Pipeline parallelism': pp,
                'Tensor parallelism': tp,
                'Phase': args.phase,
                'Total Latency (s)': total_latency,
                'Throughput (tokens/s)': average_throughput,
                'Energy per Token (mJ)': average_energy,
                'Total power (W)': total_power
            }
            new_result_df = pd.DataFrame([new_result])
            results_df = pd.concat([results_df, new_result_df], ignore_index=True)

    else:

        df = df_simulation[(df_simulation['Model'] == args.model) & (df_simulation['Pipeline parallelism'] == TransformerBlock_number[args.model]) & (df_simulation['Tensor parallelism'] == 1)]

        if args.phase == "prefill":
            df = df[(df['Sequence length'] <= args.prefill)]
            seqlen = args.prefill
        elif args.phase == "decoding":
            df = df[((args.prefill + args.decoding) >= df['Sequence length']) & (df['Sequence length'] > args.prefill)]
            seqlen = args.decoding
        elif args.phase == "end2end":
            df = df[((args.prefill + args.decoding) >= df['Sequence length'])]
            seqlen = args.prefill + args.decoding

        average_throughput = df['Throughput (tokens/s)'].mean()
        average_energy = df['Token energy (mJ)'].mean()
        total_latency = df['Token latency (ms)'].mean() * seqlen / 1000 # [Song]: convert ms to s
        total_power = df['Total power (W)'].mean()

        new_result = {
            'Model': args.model,
            'Device number': args.num_devices,
            'Seqlen': args.prefill + args.decoding,
            'Pipeline parallelism': TransformerBlock_number[args.model],
            'Tensor parallelism': 1,
            'Phase': args.phase,
            'Total Latency (s)': total_latency,
            'Throughput (tokens/s)': average_throughput,
            'Energy per Token (mJ)': average_energy,
            'Total power (W)': total_power
        }
        new_result_df = pd.DataFrame([new_result])
        results_df = pd.concat([results_df, new_result_df], ignore_index=True)
    
    results_df = results_df.drop_duplicates()
    results_df = results_df.sort_values(by=['Model', 'Device number', 'Seqlen', 'Pipeline parallelism', 'Tensor parallelism', 'Phase'])
    results_df.to_csv(args.processed_result_path, index=False)


# [Song] Simulation flow (end-to-end transformer evaluation; not single GEMV)
#
# Directory hierarchy:
#   CENT/
#   ├── cent_simulation/
#   │   ├── run_sim.py                        (this file)
#   │   ├── simulation_results.csv            (step 4 output: raw per-seqlen results)
#   │   └── processed_results.csv             (step 5 output: aggregated metrics by phase)
#   ├── trace/
#   │   ├── compile.sh / compile.py           (log parsers, copied into each mode dir at step 3)
#   │   └── {num_channels}_channels_per_device/
#   │       └── {mode}/                       (pipeline_parallel / model_parallel / *_embedding / *_FC)
#   │           └── {model}/                  (Llama2-7B / Llama2-13B / Llama2-70B) - compile dir at process_results()
#   │               ├── trace_{config}_seqlen_{N}.txt       (step 1 output: AiM ISR trace)
#   │               ├── trace_{config}_seqlen_{N}.txt.log   (step 2 output: ramulator2 raw stats)
#   │               ├── result.txt                          (step 3 intermediate: grepped stats)
#   │               └── compiled_results.txt                (step 3 output: PIM latency in ms)
#   └── figure_scripts/
#       └── figure_*.py                       (reads simulation_results.csv / processed_results.csv)
#
# 1. PIM trace generation
#    inp: per CXL device settings (FC_devices, channels_per_block, seqlen, etc.) as arguments
#    out: trace/{ch}_channels_per_device/{mode}/{model}/trace_{config}_seqlen_{N}.txt
#
# 2. AiM simulation (ramulator2)
#    inp: .txt from step 1
#    out: trace_{config}_seqlen_{N}.txt.log
#         .log contains: memory_system_cycles, idle_cycles, active_cycles, num_MAC_commands, etc.
#
# 3. process_results
#    .log -> compile.sh (grep required fields) -> result.txt
#    result.txt -> compile.py (parse) -> PIM latency (ms) + command counts -> compiled_results.txt
#    conversion: total_cycles / 2,000,000  (GDDR6 runs at 2GHz)
#    Note: CXL and PNM latency are NOT considered here. Just PIM results are considered here.
#

##### From now on, only PIM latency are considered. #####

# 4. update_csv
#    read PIM latency from compiled_results.txt
#    load_data_point(): combine PIM + CXL + PNM latency
#    compute token latency, throughput, energy, power per seqlen
#    write raw per-seqlen results to simulation_results.csv
#
# 5. process_throughputs (called via process_results.sh, NOT simulation.sh -- confusing naming)
#    read simulation_results.csv
#    filter by phase (prefill / decoding / end2end)
#    average metrics across seqlens in that phase
#    write aggregated metrics to processed_results.csv

if __name__ == "__main__":
    args = get_args()

    if args.seqlen:
        seqlen_list = args.seqlen
    else:
        # [Song]
        # list of KVcache sizes to simulate, sampled every seqlen_gap tokens
        # default example) prefill=512, decoding=3584, gap=128 -> [128, 256, ..., 4096]
        #
        # As CENT performs prefill token-by-token (like decoding), 
        # the main simulation logic itself doesn't differentiate prefill and decode;
        # only the sequence length matters.
        seqlen_list = [i * args.seqlen_gap for i in range(1, (args.prefill + args.decoding) // args.seqlen_gap + 1)]
    
    # [Song] - TODO: check the meaning of each mode.
    # trace subdirectory names, defined in utils.py:
    #   pipeline_parallel           : PP - ??
    #   pipeline_parallel_embedding : PP - output embedding (hidden -> vocab) trace
    #   model_parallel              : TP - ??
    #   model_parallel_embedding    : TP - output embedding PIM trace
    #   model_parallel_FC           : TP - ??
    for mode in pipeline_parallel_mode_list + model_parallel_mode_list:
        subprocess.run(["mkdir", "-p", f"../trace/{args.num_channels}_channels_per_device/{mode}/{args.model}"])

    # [Song]
    # generate AiM ISR trace files (.txt) for each (mode x seqlen) combination
    # Note that each seqlen represents a separate trace
    if args.generate_trace:
        generate_trace(args, seqlen_list)

    # [Song] run each trace through ramulator2 (AiM simulator), outputs simulation stats to .log files
    if args.simulate_trace:
        simulate_trace(args, seqlen_list)

    if args.process_results:
        process_results(args)
        
    if args.update_csv:
        update_csv(args, seqlen_list)
        
    if args.process_throughputs:
        process_throughputs(args)
