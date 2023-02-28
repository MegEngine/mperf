import os
import pdb
import re
import argparse

g_cls_name_parents = {}
g_thresh_modify_in_def = []
g_all_funcs = []
g_all_clss = []
g_x86_event_map = {}
g_arm_event_map = {}
INTEL_X86_CMASK_BIT = 24

def parse_if(l, in_else):
    ret = ""
    # process if else (pattern: return a if x b)
    if_cnt = l.count("if")
    if l.startswith("return "):
        tmp = l.split("return ")[1]
    else:
        tmp = l
    if if_cnt > 0:
        left, right = tmp.split("if", 1)
        cond = right.split("else", 1)[0]
        ret += "if(" + cond + ") {\n"
        ret += "return " + left + ";\n"
        ret += "} else {\n"
        # pdb.set_trace()
        ret += parse_if(right.split("else", 1)[1], True)
        ret += "}\n"
        return ret
    else:
        if in_else:
            return "return " +  tmp + ";\n" 
        else:
            if l.startswith("val ="):
                return "float " + l + ";\n"
            else:
                if "lambda" in l:
                    al = l.replace("return ", "")
                    tail = al[re.search("lambda", al).span()[0]:]
                    tail = tail.strip()
                    left, right = tail.split(" : ")
                    ret = "auto func = [&]() -> float { return " + right[0:-9] + ";\n};\n"
                    ret += "return g_ev_process(func, level);\n"
                    return ret
                else:
                    return l + ";\n"

def parse_func(l, flines, idx):
    ext_param = ", bool& thresh"
    # for xl in flines[idx:]:
    #     xl = xl.strip()
    #     if xl.startswith("self.thresh ="):
    #         ext_param = ", bool& thresh"
    #         g_thresh_modify_in_def.append(l.split("def ")[1][:-1].split("(")[0])
    #         break
    #     elif xl.startswith("return"):
    #         break
    
    # process func body
    sfunc = "float " + l.split("def ")[1][:-1] + "{\n"
    sfunc = sfunc.replace("self, EV, level", "FEV EV, int level" + ext_param)
    g_all_funcs.append(sfunc.replace("{\n", ";"))
    for xl in flines[idx:]:
        idx = idx + 1
        xl = xl.strip()
        xl = xl.split("#")[0]
        if not xl:
            continue
        sfunc += parse_if(xl, False)
        if xl.startswith("return"):
            break
    sfunc += "}\n"
    sfunc = sfunc.replace("self, ", "")
    sfunc = sfunc.replace("self.", "")
    sfunc = sfunc.replace(".compute", "::compute")
    sfunc = sfunc.replace(" and ", " && ")
    sfunc = sfunc.replace(" False", " false")
    sfunc = sfunc.replace(" True", " true")
    sfunc = sfunc.replace(" min(", " std::min<float>(")
    sfunc = sfunc.replace(" max(", " std::max<float>(")
    sfunc = sfunc.replace("(EV, level)", "(EV, level, thresh)")
    pat = re.compile("\(EV, \d\)")
    ret = pat.findall(sfunc)
    for str in ret:
        sfunc = sfunc.replace(str, str[:-1] + " , thresh)")

    return sfunc, idx

def parse_desc(l, flines, idx):
    desc = ""
    for xl in flines[idx:]:
        idx = idx + 1
        xl = xl.strip()
        if xl.endswith("\"\"\""):
            desc += xl.split("\"\"\"")[0]
            break
        else:
            desc += xl + " "
    return desc,idx

def parse_class(l, flines, idx):
    nl = l.split(" ")
    cls_name = nl[1].split(":")[0]
    domain = ""
    area = ""
    level = ""
    desc = ""
    compute = ""
    thresh = ""
    for xl in flines[idx:]:
        xl = xl.strip()
        xl = xl.split("#")[0]
        idx = idx + 1
        if xl.startswith("desc"):
            # pdb.set_trace()
            desc,idx = parse_desc(xl, flines, idx)
            break
        if xl.startswith("class"):
            idx = idx -1 # process knl_ratios.py
            break
        elif xl.startswith("domain"):
            domain = xl.split("=")[1]
        elif xl.startswith("area"):
            area = xl.split("=")[1]        
        elif xl.startswith("level"):
            level = xl.split("=")[1]    
        elif xl.startswith("self.val"):
            if "self.val" in xl.split("=")[1]:
                compute += ";\n" + xl.replace("self.", "")
                continue
            compute = xl.split("=")[1] 
            compute = compute.replace("self, ", "")
            compute = compute.replace("self.", "")
            compute = compute.replace(".compute", "::compute")
        elif xl.startswith("self.thresh"):
            thresh = xl.split("=")[1] 
            pname = ""
            if cls_name in g_cls_name_parents:
                pname = g_cls_name_parents[cls_name]
                # print(cls_name, pname)
            thresh = thresh.replace("and self.parent.thresh", "&& " + pname + "::thresh")
            thresh = thresh.replace(" and ", " && ")
            thresh = thresh.replace(" or ", " | ")
            thresh = thresh.replace("self, ", "")
            thresh = thresh.replace("self.val", "val")
            thresh = thresh.replace("self.", "")
            thresh = thresh.replace(".thresh", "::thresh")
            thresh = thresh.replace(" False", " false")
            thresh = thresh.replace(" True", " true")
        else:
            continue
    
    g_all_clss.append("class " + cls_name + ";")
    data_compute_def = ""
    data = "class " + cls_name + " : public MetricBase {\n"
    data += "public: \n"
    data += cls_name + "() {\n"
    data += "name = \"" + cls_name + "\";\n"
    data += "domain = " + domain + ";\n"
    data += "area = " + area + ";\n"
    if level:
        data += "level = " + level + ";\n"
    data += "desc = \"" + desc + "\";\n"
    data += "parent = nullptr;\n"
    data += "func_compute = (void*)&compute;\n"

    data += "}\n"

    data += "static float compute(FEV EV);\n"
    data += "};\n"

    data_compute_def += "float " + cls_name + "::compute(FEV EV) {\n"
    data_compute_def += "val = " + compute + ";\n"
    data_compute_def += "thresh = " + thresh + ";\n"
    data_compute_def += "return val;\n"
    data_compute_def += "}\n"

    data_compute_def = data_compute_def.replace(" min(", " std::min<float>(")
    data_compute_def = data_compute_def.replace(" max(", " std::max<float>(")
    # print(data)
    # pdb.set_trace()
    return data, data_compute_def, idx

    
def parse_expr(l, data):
    if "=" not in l:
        return ""
    if len(re.findall("=", l)) > 2:
        return "" # PMM_App_Direct = 1 if Memory == 1 else 0
    left, right = l.split("=")
    float_pat = re.compile(r'\d+\.?\d+')
    int_pat = re.compile(r'\d')
    left = left.strip()
    right = right.strip()

    # special process
    if "class " + left + "(" in data:
        return ""
    if "def " + left + "(" in data:
        return ""

    # process expr
    if right.startswith("lambda"):
        return ""
    elif right.startswith("False"):
        return "static bool " + left + " = " + "false;\n"
    elif right.startswith("True"):
        return "static bool " + left + " = " + "true;\n"
    elif left.startswith("version"):
        return "static std::string version = " + right + ";\n"
    elif float_pat.findall(right):
        return "static float " + left + " = " + right + ";\n"
    elif int_pat.findall(right):
        return "static float " + left + " = " + right + ";\n"
    else:
        return ""

def parse_parents(flines):
    for nl in flines:
        nl = nl.strip()
        if ".parent = " in nl:
            key,val = nl.split(".parent = ")
            key = key.split("[")[1].split("]")[0]
            val = val.split("[")[1].split("]")[0]
            g_cls_name_parents[key[1:-1]] = val[1:-1]
flines = []
str_cls_def = ""
str_cls_declare = ""
str_def = ""
str_expr = ""
def parse_ratios_file(is_x86, map_file, in_name, ot_name, arch_cpu_name):
    global flines
    global str_cls_def
    global str_cls_declare
    global str_def
    global str_expr

    data = ""
    with open(in_name, "r") as f:
        data = f.read()

    with open(in_name, "r") as f:
        flines = f.readlines()
        
    flines = list(line for line in flines if line)

    # get parent
    cur_idx = 0
    for l in flines:
        cur_idx = cur_idx + 1
        l = l.strip()
        if l.startswith("class Setup:"):
            parse_parents(flines[cur_idx:])


    cur_idx = 0
    for n in range(len(flines)):
        n += cur_idx
        l = flines[cur_idx]
        l = l.strip()
        cur_idx = cur_idx + 1
        if l.startswith("#") or l.startswith("import"):
            continue
        elif l.startswith("def"):
            if l.startswith("def handle_error("):
                cur_idx = cur_idx + 4
                continue
            elif l.startswith("def handle_error_metric("):
                cur_idx = cur_idx + 3
                continue
            tmp, cur_idx = parse_func(l, flines, cur_idx)
            str_def += tmp
        elif l.startswith("class"):
            # process class
            nl = l.split(" ")
            cls_name = nl[1].split(":")[0]
            if cls_name.startswith("Setup"):
                break
            cls_tmp, cls_tmp_def, cur_idx = parse_class(l, flines, cur_idx)
            str_cls_declare += cls_tmp
            str_cls_def += cls_tmp_def
        else:
            str_expr += parse_expr(l, data)

    # process modify self.thesh
    pat = re.compile("\(EV, \d\)")
    ret = pat.findall(str_cls_def)
    for sr in ret:
        str_cls_def = str_cls_def.replace(sr, sr[:-1] + " , thresh)")

    if is_x86:
        with open(map_file, "r") as mf:
            lines = mf.readlines()
            for line in lines:
                line = line.strip()
                k = line.split(',', 1)[0]
                v = line.split(',', 1)[1]
                g_x86_event_map[k] = v

        str_cls_def = str_cls_def.replace('EV("', 'g_ev_error("')
        str_def = str_def.replace('EV("', 'g_ev_error("')

        for k,v in g_x86_event_map.items():
            av = v.split(",")
            lav = len(av)
            config = 0
            exclude_user = 0
            if lav == 2: # cpu/event=,umask=
                if av[0].split("=")[0].strip() != "cpu/event":
                    continue
                if av[1].split("=")[0].strip() != "umask":
                    continue
                ecode = int(av[0].split("=")[1].strip(), 0)
                umask = int(av[1].split("=")[1].strip().split("/")[0], 0)
                if av[1].split("=")[1].strip().split("/")[1] == 'k':
                    exclude_user = 1 # only count kernel
                config = hex(ecode | (umask << 8))
            elif lav == 3:  # cpu/event=,umask=,cmask=
                if av[0].split("=")[0].strip() != "cpu/event":
                    continue
                if av[1].split("=")[0].strip() != "umask":
                    continue
                if av[2].split("=")[0].strip() != "cmask":
                    continue
                ecode = int(av[0].split("=")[1].strip(), 0)
                umask = int(av[1].split("=")[1].strip().split("/")[0], 0)
                cmask = int(av[2].split("=")[1].strip()[0:-1], 0)
                # print(av, ecode, umask, cmask)
                config = hex(ecode | (umask << 8) | (cmask << INTEL_X86_CMASK_BIT))
            else:
                config = 0
                exclude_user = 0

            # print(str(rv))
            rp_s = "g_ev_error(\"" + k + "\""
            # PERF_TYPE_RAW: 4
            rp_d = "EV({\"" + k + "\", " + str(config) + ", 4, " + str(exclude_user) + "}"
            str_cls_def = str_cls_def.replace(rp_s, rp_d)
            str_def = str_def.replace(rp_s, rp_d)
    else:
        arm_event_name = []
        arm_value_name = []

        print("-----------------")
        with open(map_file, "r") as mf:
            lines = mf.readlines()
            for line in lines:
                line = line.strip()
                if ".name" in line:
                    arm_event_name.append(line.split("=")[1][:-1].strip())
                elif ".code" in line:
                    arm_value_name.append(line.split("=")[1][:-1].strip())
                
        alen = len(arm_event_name)
        for ia in range(alen):
            g_arm_event_map[arm_event_name[ia]] = arm_value_name[ia]
        
        str_cls_def = str_cls_def.replace('EV("', 'g_ev_error("')
        str_def = str_def.replace('EV("', 'g_ev_error("')
        
        for k,v in g_arm_event_map.items():
            rp_s = "g_ev_error(" + k
            rp_d = "EV({" + k + ", " + str(v) + ", 4, 0}"
            str_cls_def = str_cls_def.replace(rp_s, rp_d)
            str_def = str_def.replace(rp_s, rp_d)

    # print output
    with open(ot_name, "w") as f:
        f.write("// Automatically generated by the script cvt_from_pmu_tools.sh.\n")
        f.write("#include \"arch_ratios.h\"\n")
        f.write('\n')
        f.write('namespace mperf {\n')
        f.write('namespace tma {\n')
        f.write('\n')

        # expr
        f.write(str_expr)
        f.write('\n\n')

        # func decl
        for nf in g_all_funcs:
            f.write(nf)
        f.write('\n\n')

        # class def
        f.write(str_cls_declare)
        f.write('\n\n')
        f.write(str_cls_def)
        f.write('\n\n')

        # func def
        f.write(str_def)
        f.write('\n\n')

        # SetUp
        clsname = arch_cpu_name.upper() + "SetUpImpl"
        setup_cls = clsname + "::" + clsname + "(){\n"
        for nc in g_all_clss:
            nc = nc.replace("class", "").strip()
            nc = nc[:-1]
            if nc.startswith("Metric"):
                setup_cls += "m_vmtc_extra.push_back(std::make_pair(\"" + nc + "\", (MetricBase*)(new " + nc + "())));\n"
            else:
                setup_cls += "m_vmtc_core.push_back(std::make_pair(\"" + nc + "\", (MetricBase*)(new " + nc + "())));\n"
        setup_cls += "}\n"
        
        setup_cls += clsname + "::~" + clsname + "(){\n"
        setup_cls += "size_t cz = m_vmtc_core.size();\n"
        setup_cls += "for(size_t i=0; i<cz; ++i){\n"
        setup_cls += "if(m_vmtc_core[i].second) {\n"
        setup_cls += "delete m_vmtc_core[i].second;\n"
        setup_cls += "}}\n"

        setup_cls += "size_t ez = m_vmtc_extra.size();"
        setup_cls += "for(size_t i=0; i<ez; ++i){"
        setup_cls += "if(m_vmtc_extra[i].second) {"
        setup_cls += "delete m_vmtc_extra[i].second;"
        setup_cls += "}}\n"
        setup_cls += "}\n"

        f.write(setup_cls)
        f.write('\n\n')


        f.write('}  // namespace tma\n')
        f.write('}  // namespace mperf')


def main():
    parser = argparse.ArgumentParser(
        description='convert pmu-tools arch-pmu-file from python to cpp',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-T",
        "--target",
        type=str,
        help="x86 or arm",
    )

    parser.add_argument(
        "-I",
        "--input-file",
        type=str,
        help="choose arch-pmu-file",
    )

    parser.add_argument(
        "-M",
        "--map-file",
        type=str,
        help="input event map file",
    )

    parser.add_argument(
        "-O",
        "--output-file",
        type=str,
        help="output filename",
    )

    args = parser.parse_args()
    print("=============================")
    print("input file: ", args.input_file)
    print("map file: ",args.map_file)
    print("output file: ",args.output_file)

    is_x86 = True
    if args.target == "x86":
        is_x86 = True
    else:
        is_x86 = False

    arch_cpu_name = args.input_file.split("/")[-1].split("_ratios.py")[0].replace("_", "")
    
    parse_ratios_file(is_x86, args.map_file, args.input_file, args.output_file, arch_cpu_name)

main()