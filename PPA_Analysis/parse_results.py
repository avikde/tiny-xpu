import re
from pathlib import Path
import csv

csv_path = "results/ppa_data.csv"
csv_data = [["Rows/Columns", "Data Width", "Accumulator Width", "Total Area", "Sequential Area", "Critical Path Time", "Internal Power", "Switching Power", "Leakage Power", "Total Power"]]
for DW in [4, 8, 16, 32]:
    for AW in [16, 32, 64, 128]:
        for N in [4, 8, 16, 32]:

            print(f"Configuration: {N}_{N}_{DW}_{AW}")

            total_area = None
            seq_area = None
            data_arr_time = None
            internal_power = None
            switching_power = None
            leakage_power = None
            total_power = None

            folder_name = f"results/R{N}_C{N}_DW{DW}_AW{AW}/"
            synth_log = folder_name + "SYNTH.log"
            timing_log = folder_name + "TIMING.log"
            power_log = folder_name + "POWER.log"

            # SYNTH.log
            if not Path(synth_log).exists():
                synth_log_str = ''
            else:
                with open (synth_log, "r") as f:
                    synth_log_str = f.read()
            m = re.search(r"Chip area for top module.*:\s([\d.]+)", synth_log_str)
            if m:
                total_area = float(m.group(1))
            m = re.findall(r"of which used for sequential elements.*:\s([\d.]+)", synth_log_str)
            if m:
                seq_area = float(m[-1])

            # TIMING.log
            if not Path(timing_log).exists():
                timing_log_str = ''
            else:
                with open (timing_log, "r") as f:
                    timing_log_str = f.read()
            m = re.search(r"([\d.]+)\s+data arrival time", timing_log_str)
            if m:
                data_arr_time = float(m.group(1))

            # POWER.log
            if not Path(power_log).exists():
                power_log_str = ''
            else:
                with open (power_log, "r") as f:
                    power_log_str = f.read()
            m = re.search(
            r"^Total\s+"
            r"(\S+)\s+"      # Internal
            r"(\S+)\s+"      # Switching
            r"(\S+)\s+"      # Leakage
            r"(\S+)",        # Total
            power_log_str,
            re.MULTILINE
            )
            if m:
                internal_power = float(m.group(1))
                switching_power = float(m.group(2))
                leakage_power = float(m.group(3))
                total_power = float(m.group(4))

            print(f"Total Area:  {total_area}")
            print(f"Sequential Area: {seq_area}")
            print(f"Data Arrival Time: {data_arr_time}")
            print(f"Internal Power: {internal_power}")
            print(f"Switching Power: {switching_power}")
            print(f"Leakage Power: {leakage_power}")
            print(f"Total Power: {total_power}")
            print("----------------")

            csv_data.append([N, DW, AW, total_area, seq_area, data_arr_time, internal_power, switching_power, leakage_power, total_power])
with open(csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)
