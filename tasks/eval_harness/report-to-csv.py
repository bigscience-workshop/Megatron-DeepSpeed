#!/usr/bin/env python

# this script converts results.json:
#
#   "results": {
#     "arc_challenge": {
#       "acc": 0.24232081911262798,
#       "acc_stderr": 0.01252159329580012,
#       "acc_norm": 0.2764505119453925,
#       "acc_norm_stderr": 0.013069662474252425
#     },
#
# into a format expected by a spreadsheet, which is:
#
#   task          metric   value    err
#   arc_challenge acc      xxx      yyy
#   arc_challenge acc_norm xxx      yyy
#   arc_challenge f1       xxx      yyy
#
# usage:
# report-to-csv.py results.json


import sys
import json
import io
import csv

results_file = sys.argv[1]

csv_file = results_file.replace("json", "csv")

print(f"Converting {results_file} to {csv_file}")

with io.open(results_file, 'r', encoding='utf-8') as f:
    results = json.load(f)

with io.open(csv_file, 'w', encoding='utf-8') as f:

    writer = csv.writer(f)
    writer.writerow(["task", "metric", "value", "err", "version"])

    versions = results["versions"]

    for k,v in sorted(results["results"].items()):
        if "acc" in v:
            row = [k, "acc", v["acc"], v["acc_stderr"]]
        if "acc_norm" in v:
            row = [k, "acc_norm", v["acc_norm"], v["acc_norm_stderr"]]
        if "f1" in v:
            row = [k, "f1", v["f1"], v["f1_stderr"] if "f1_stderr" in v else ""]
        # if "ppl" in v:
        #     row = [k, "ppl", v["ppl"], v["ppl_stderr"]]
        # if "em" in v:
        #     row = [k, "em", v["em"], v["em_stderr"] if "em_stderr" in v else ""]

        row += [versions[k] if k in versions else -1]
        writer.writerow(row)
