import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("datacard")
args = parser.parse_args()

with open(args.datacard) as f:
    datacard = f.readlines()

new_process_names = ["EWK_TT_lin_pos", "EWK_TT_lin_neg", "EWK_TT_quad_pos", "EWK_TT_quad_neg"]
new_process_idx = [-6, -7, -8, -9]

new_datacard = []
had_first_process_line = False
had_first_bin_line = False
num_seperators = 0
for l in datacard:
    l = l.strip()
    if l.startswith("---"):
        num_seperators += 1
    
    elif l.startswith("imax"):
        nbins = int(re.findall(r"\d+", l)[0])

    elif l.startswith("jmax"):
        current_num_proc = re.findall(r"\d+", l)[0]
        nproc_old = int(current_num_proc) + 1
        nproc_new = nproc_old + len(new_process_names)
        l = l.replace(current_num_proc, str(nproc_new - 1))

    elif l.startswith("kmax"):
        current_nuisances = re.findall(r"\d+", l)[0]
        l = l.replace(current_nuisances, str(int(current_nuisances) - 1))

    elif l.startswith("bin"):
        if not had_first_bin_line:
            had_first_bin_line = True
        else:
            bins = re.sub(' +', ' ', l.strip()).split(" ")[1:]
            #assert len(bins) == nproc_old * nbins
            unique_bins = []
            for bin in bins:
                if not bin in unique_bins:
                    unique_bins.append(bin)
            
            
            new_bins = []
            for bin in unique_bins:
                for i in range(len(new_process_names)):
                    new_bins.append(bin)
                    
            l += " " + " ".join(new_bins)

    elif l.startswith("process"):
        if not had_first_process_line:
            new_processes = []
            for bin in unique_bins:
                for i in range(len(new_process_names)):
                    new_processes.append(new_process_names[i])

            l += " " + " ".join(new_processes)
            had_first_process_line = True
        else:
            new_idx = []
            for bin in unique_bins:
                for i in range(len(new_process_idx)):
                    new_idx.append(str(new_process_idx[i]))

            l += " " + " ".join(new_idx)

    elif l.startswith("rate"):
        l += " -1" * (nbins * len(new_process_names))

    elif num_seperators == 4:
        if l.startswith("EWK_yukawa"):
            l = ""
        elif ("shape" in l or "lnN" in l) and not ("group" in l):
            l += " -" * (nbins * len(new_process_names))
        elif "group" in l:
            l = l.replace(" EWK_yukawa", "")

    new_datacard.append(l)

new_datacard = "\n".join(new_datacard)

with open(args.datacard, "w") as f:
    f.write(new_datacard)