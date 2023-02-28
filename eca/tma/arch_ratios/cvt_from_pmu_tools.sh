#!/bin/bash -e

PMU_TOOS=/path/pmu-tools
DST_PATH=.

echo $PMU_TOOS
RATIOS_FILES=$(ls ${PMU_TOOS}/*ratios.py | rev | cut -d/ -f 1 | rev)

# generate x86 events map file
/path/pmu-tools/toplev.py -l6 -S -v  -m --no-desc --no-multiplex --force-cpu hsw  --dump-raw-events 1 --dump-raw-events-file ./raw_map.txt


# process x86 ratios files
for IN_FNAME in $RATIOS_FILES
do
    if [[ $IN_FNAME = "knl_ratios.py" ]] || [[ $IN_FNAME = "simple_ratios.py" ]] || [[ $IN_FNAME = "slm_ratios.py" ]]
    then
        echo "skip $IN_FNAME"
        continue
    else
        OT_FNAME=$(echo ${IN_FNAME} | cut -d. -f 1).cpp
        IN_NAME=$PMU_TOOS/$IN_FNAME
        OT_NAME=$DST_PATH/$OT_FNAME
        python3 ./gen_from_pmu_tools.py -T x86 -I $IN_NAME -M ./raw_map.txt -O $OT_NAME
        clang-format-5.0 -i $OT_NAME
    fi
done

# process arm ratios files
python3 ./gen_from_pmu_tools.py -T arm -I $DST_PATH/a55_ratios.py -M ../../../../../third_party/libpfm4/lib/events/arm_cortex_a55_events.h -O $DST_PATH/a55_ratios.cpp
clang-format-5.0 -i $DST_PATH/a55_ratios.cpp