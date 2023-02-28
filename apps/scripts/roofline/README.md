## apps/scripts/roofline
* The raw version of plot_roofline.py and plot_roofline_hierarchical.py come from [nersc-roofline](https://github.com/cyanguwa/nersc-roofline.git), and mperf have some modifications.
* the input data for plot_roofline.py is commented as follows:
    ```bash
    # some comments about the spcific platform

    # params for plotting roofs
    memroofs 26.3            # the measured peak memory bandwidth on the platform, unit: GBPs
    mem_roof_names 'DRAM'    # the name of the memory hierarchy, such as 'L1' 'L2' 'L3' or "DRAM"
    comproofs 1159           # the measured peak compute bandwidth on the platform, unit: GFLOPs
    comp_roof_names 'FMA'    # the name of the instruction type, such as 'FMA' 'NO-FMA'

    # omit the following if only plotting roofs
    # params for plotting your measured function as a point in the graph
    AI 15.5                  # the arithmetic intensity of your measured function, unit: flops_per_byte
    FLOPS 261                # the compute bandwidth of your measured function, unit: GFLOPs
    labels 'FMA, DRAM'       # legend label of the plot point
    ```
* the input data for plot_roofline_hierarchical.py is commented as follows:
    ```bash
    # some comments about the spcific platform

    # params for plotting roofs
    memroofs 11.9 10.0 4.6          # the measured peak memory bandwidth about different memory hierarchies on the platform, unit: GBPs
    mem_roof_names 'L2' 'L3' 'DRAM' # the names of the memory hierarchies
    comproofs 12.3 6.1              # the measured peak compute bandwidth about different instruction type on the platform, unit: GFLOPs
    comp_roof_names 'FMA' 'No-FMA'  # the names of the instruction type

    # omit the following if only plotting roofs
    # params for plotting your measured function as points in the graph
    AI_L2 1.645                     # the arithmetic intensity of your measured function in L2 cache hierarchy, unit: flops_per_byte
    AI_L3 2.587                     # the arithmetic intensity of your measured function in L3 cache hierarchy, unit: flops_per_byte
    AI_DRAM 3.396                   # the arithmetic intensity of your measured function in dram hierarchy, unit: flops_per_byte
    FLOPS 1.63 2.07 1.53            # the measured peak compute bandwidth about different problem size on the platform, unit: GFLOPs 
    labels 'gaussian_blur_ksize9'   # legend label of the plot points
    ```
* plot the roofline using the command:
    ```bash
    python3 plot_roofline.py ./roofline_data.txt
    or
    python3 plot_roofline_hierarchical.py ./roofline_data_hierarchical.txt
    ```
