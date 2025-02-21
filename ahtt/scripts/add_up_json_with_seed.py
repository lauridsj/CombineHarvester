import json
import argparse
import glob
from collections import defaultdict

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def aggregate_json(files, seed_file=None):
    aggregated = {
        "points": [],
        "best_fit_g1_g2_dnll": [],
        "g-grid": defaultdict(lambda: {"total": 0, "pass": 0, "dnll": 0.0})
    }
    seed_data = None
    
    if seed_file:
        seed_data = load_json(seed_file)
        aggregated["points"] = seed_data.get("points", [])
        aggregated["best_fit_g1_g2_dnll"] = seed_data.get("best_fit_g1_g2_dnll", [])
        seed_grid = seed_data.get("g-grid", {})
    else:
        seed_grid = {}
    
    for file in files:
        data = load_json(file)
        if "points" in data:
            aggregated["points"].extend(p for p in data["points"] if p not in aggregated["points"])
        
        for key, values in data.get("g-grid", {}).items():
            total, passed, dnll = values["total"], values["pass"], values["dnll"]
            if seed_data and key in seed_grid:
                total -= seed_grid[key]["total"]
                passed -= seed_grid[key]["pass"]
            aggregated["g-grid"][key]["total"] += total
            aggregated["g-grid"][key]["pass"] += passed
    
    if seed_data:
        for key, values in seed_grid.items():
            aggregated["g-grid"][key]["dnll"] += values["dnll"]
            aggregated["g-grid"][key]["total"] += values["total"]
            aggregated["g-grid"][key]["pass"] += values["pass"]
    
    return aggregated

def main():
    parser = argparse.ArgumentParser(description="Aggregate JSON files with optional seed subtraction.")
    parser.add_argument("files", nargs='+', help="Input JSON files (can use wildcards)")
    parser.add_argument("--seed", help="Seed JSON file to subtract before summing")
    parser.add_argument("--output", default="aggregated.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    file_list = []
    for pattern in args.files:
        file_list.extend(glob.glob(pattern))
    
    aggregated_data = aggregate_json(file_list, args.seed)
    
    with open(args.output, "w") as f:
        json.dump(aggregated_data, f, indent=2)
    
    print(f"Aggregated data saved to {args.output}")

if __name__ == "__main__":
    main()

