import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def parse_bw_mem(fname):
    data = {}
    lines = None
    with open(fname, "r") as f:
        lines = f.readlines()
    first_item = lines[0].split()[0]
    data["size"] = []
    for line in lines:
        ln_data = line.split()
        if first_item != ln_data[0]:
            break
        data["size"].append((ln_data[1]))
    
    for line in lines:
        ln_data = line.split()
        if ln_data[0] not in data:
            data[ln_data[0]] = []
        data[ln_data[0]].append((float)(ln_data[2])/1024) # MBPS->GBPS
    return data

def main():
    parser = argparse.ArgumentParser(
        description='generate header file from midout traces',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--filename', help='input file name')
    parser.add_argument('-c', '--core', help='core arch')
    args = parser.parse_args()
    fname = args.filename
    core = args.core
    data = parse_bw_mem(fname)
    print(pd.DataFrame(data))

    axis_sz = data['size']
    # style = ['color=b.-', 'g.-', 'r.-', 'c.-', 'm.-', 'y.-', 'k.-', 'm.-', 'tan.-', 'teal.-', 'gray.-']
    idx = 0
    max_val = 0
    for key, val in data.items():
        if key != 'size':
            plt.plot(axis_sz, val, '.-', label=key)
            idx = idx + 1
            if max(val) > max_val:
                max_val = max(val)
    len_sz = len(axis_sz)
    # print(axis_sz)
    L1_LINE = ['65536' for x in range(0, len_sz)]
    L2_128K_LINE = ['131072' for x in range(0, len_sz)]
    L2_256K_LINE = ['262144' for x in range(0, len_sz)]
    L2_512K_LINE = ['524288' for x in range(0, len_sz)]
    L3_2M_LINE = ['2097152' for x in range(0, len_sz)]
    L3_4M_LINE = ['4194304' for x in range(0, len_sz)]
    Y_LINE = [max_val for x in range(0, len_sz)]
    Y_LINE[0] = 0
    plt.plot(L1_LINE, Y_LINE, 'k--', label="L1(64K)", linewidth=0.5)
    plt.plot(L2_128K_LINE, Y_LINE, 'k--', label="L2(128K)", linewidth=0.5)
    plt.plot(L2_256K_LINE, Y_LINE, 'k--', label="L2(256K)", linewidth=0.5)
    plt.plot(L2_512K_LINE, Y_LINE, 'k--', label="L2(512K)", linewidth=0.5)
    plt.plot(L3_2M_LINE, Y_LINE, 'k--', label="L3(2M)", linewidth=0.5)
    plt.plot(L3_4M_LINE, Y_LINE, 'k--', label="L3(4M)", linewidth=0.5)

    plt.title(core)
    plt.legend()
    plt.xlabel('size')
    plt.ylabel('GBPS')
    plt.show()

main()
