# Introduction

This notebook is written to demonstrate the use of M3GNet as a structure relaxer as well as to provide more comprehensive benchmarks for cubic crystals based on exp data on Wikipedia and MP DFT data. This benchmark is limited to cubic crystals for ease of comparison since there is only one lattice parameter. 

If you are running this notebook from Google Colab, uncomment the next code box to install matgl first.


```python
# !pip install matgl
```


```python
from __future__ import annotations

import traceback
import warnings

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Lattice, Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm

import matgl
from matgl.ext.ase import Relaxer

warnings.filterwarnings("ignore")
```

The next cell just compiles data from Wikipedia. 


```python
data = pd.read_html("http://en.wikipedia.org/wiki/Lattice_constant")[0]
data = data[
    ~data["Crystal structure"].isin(
        ["Hexagonal", "Wurtzite", "Wurtzite (HCP)", "Orthorhombic", "Tetragonal perovskite", "Orthorhombic perovskite"]
    )
]
data = data.rename(columns={"Lattice constant (Å)": "a (Å)"})
data = data.drop(columns=["Ref."])
data["a (Å)"] = data["a (Å)"].map(float)
data = data[["Material", "Crystal structure", "a (Å)"]]


additional_fcc = """10 Ne 4.43 54 Xe 6.20
13 Al 4.05 58 Ce 5.16
18 Ar 5.26 70 Yb 5.49
20 Ca 5.58 77 Ir 3.84
28 Ni 3.52 78 Pt 3.92
29 Cu 3.61 79 Au 4.08
36 Kr 5.72 82 Pb 4.95
38 Sr 6.08 47 Ag 4.09
45 Rh 3.80 89 Ac 5.31
46 Pd 3.89 90 Th 5.08"""

additional_bcc = """3 Li 3.49 42 Mo 3.15
11 Na 4.23 55 Cs 6.05
19 K 5.23 56 Ba 5.02
23 V 3.02 63 Eu 4.61
24 Cr 2.88 73 Ta 3.31
26 Fe 2.87 74 W 3.16
37 Rb 5.59 41 Nb 3.30"""


def add_new(str_, structure_type, df):
    tokens = str_.split()
    new_crystals = []
    for i in range(int(len(tokens) / 3)):
        el = tokens[3 * i + 1].strip()
        if el not in df["Material"].to_numpy():
            new_crystals.append([tokens[3 * i + 1], structure_type, float(tokens[3 * i + 2])])
    df2 = pd.DataFrame(new_crystals, columns=data.columns)
    return pd.concat([df, df2])


data = add_new(additional_fcc, "FCC", data)
data = add_new(additional_bcc, "BCC", data)
data = data[data["Material"] != "NC0.99"]
data = data[data["Material"] != "Xe"]
data = data[data["Material"] != "Kr"]
data = data[data["Material"] != "Rb"]
data = data.set_index("Material")
print(data)
```

                 Crystal structure     a (Å)
    Material                                
    C (diamond)      Diamond (FCC)  3.567000
    Si               Diamond (FCC)  5.431021
    Ge               Diamond (FCC)  5.658000
    AlAs         Zinc blende (FCC)  5.660500
    AlP          Zinc blende (FCC)  5.451000
    ...                        ...       ...
    Cs                         BCC  6.050000
    K                          BCC  5.230000
    Ba                         BCC  5.020000
    Eu                         BCC  4.610000
    Cr                         BCC  2.880000
    
    [89 rows x 2 columns]


In the next cell, we generate an initial structure for all the phases. The cubic constant is set to an arbitrary value of 5 angstroms for all structures. It does not matter too much what you set it to, but it cannot be too large or it will result in isolated atoms due to the cutoffs used in m3gnet to determine bonds. We then call the Relaxer, which is the M3GNet universal IAP pre-trained on the Materials Project.


```python
predicted = []
mp = []
mpr = MPRester()

# Load the pre-trained M3GNet Potential
pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")

# create the M3GNet Relaxer
relaxer = Relaxer(potential=pot)

for formula, v in tqdm(data.iterrows(), total=len(data)):
    formula = formula.split()[0]
    c = Composition(formula)
    els = sorted(c.elements)
    cs = v["Crystal structure"]

    # We initialize all the crystals with an arbitrary lattice constant of 5 angstroms.
    if "Zinc blende" in cs:
        s = Structure.from_spacegroup("F-43m", Lattice.cubic(4.5), [els[0], els[1]], [[0, 0, 0], [0.25, 0.25, 0.75]])
    elif "Halite" in cs:
        s = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.5), [els[0], els[1]], [[0, 0, 0], [0.5, 0, 0]])
    elif "Caesium chloride" in cs:
        s = Structure.from_spacegroup("Pm-3m", Lattice.cubic(4.5), [els[0], els[1]], [[0, 0, 0], [0.5, 0.5, 0.5]])
    elif "Cubic perovskite" in cs:
        s = Structure(
            Lattice.cubic(5),
            [els[0], els[1], els[2], els[2], els[2]],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.0, 0.5, 0.5], [0.5, 0, 0.5]],
        )
    elif "Diamond" in cs:
        s = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5), [els[0]], [[0.25, 0.75, 0.25]])
    elif "BCC" in cs:
        s = Structure(Lattice.cubic(4.5), [els[0]] * 2, [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    elif "FCC" in cs:
        s = Structure(Lattice.cubic(4.5), [els[0]] * 4, [[0.0, 0.0, 0.0], [0.5, 0.5, 0], [0.0, 0.5, 0.5], [0.5, 0, 0.5]])
    else:
        predicted.append(0)
        mp.append(0)
        continue

    # print(s.composition.reduced_formula)
    relax_results = relaxer.relax(s, fmax=0.01)

    final_structure = relax_results["final_structure"]

    predicted.append(final_structure.lattice.a)

    try:
        mids = mpr.get_material_ids(s.composition.reduced_formula)
        for i in mids:
            try:
                structure = mpr.get_structure_by_material_id(i)
                sga = SpacegroupAnalyzer(structure)
                sga2 = SpacegroupAnalyzer(final_structure)
                if sga.get_space_group_number() == sga2.get_space_group_number():
                    conv = sga.get_conventional_standard_structure()
                    mp.append(conv.lattice.a)
                    break
            except Exception:
                pass
        else:
            raise RuntimeError
    except Exception:
        mp.append(0)
        traceback.print_exc()

data["MP a (Å)"] = mp
data["Predicted a (Å)"] = predicted
```

      0%|                                                                                                 | 0/89 [00:00<?, ?it/s]


    Retrieving MaterialsDoc documents:   0%|          | 0/62 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


      1%|█                                                                                        | 1/89 [00:03<04:55,  3.36s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/42 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


      2%|██                                                                                       | 2/89 [00:05<03:59,  2.75s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/17 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


      3%|███                                                                                      | 3/89 [00:07<03:21,  2.35s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/6 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


      4%|████                                                                                     | 4/89 [00:17<07:25,  5.24s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/3 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


      6%|█████                                                                                    | 5/89 [00:19<05:40,  4.06s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/5 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


      7%|██████                                                                                   | 6/89 [00:20<04:30,  3.26s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/6 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


      8%|███████                                                                                  | 7/89 [00:22<03:53,  2.85s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/13 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


      9%|████████                                                                                 | 8/89 [00:25<03:44,  2.78s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/9 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     10%|█████████                                                                                | 9/89 [00:27<03:24,  2.55s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/3 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     11%|█████████▉                                                                              | 10/89 [00:41<07:51,  5.97s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/5 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     12%|██████████▉                                                                             | 11/89 [00:44<06:50,  5.26s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/11 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     13%|███████████▊                                                                            | 12/89 [00:47<05:36,  4.37s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/23 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     15%|████████████▊                                                                           | 13/89 [00:49<04:42,  3.71s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/5 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     16%|█████████████▊                                                                          | 14/89 [00:51<04:07,  3.30s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/3 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     17%|██████████████▊                                                                         | 15/89 [00:55<04:22,  3.55s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/9 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     18%|███████████████▊                                                                        | 16/89 [00:57<03:34,  2.93s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/12 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     19%|████████████████▊                                                                       | 17/89 [01:01<03:47,  3.15s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/146 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     20%|█████████████████▊                                                                      | 18/89 [01:03<03:19,  2.80s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/19 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


    Traceback (most recent call last):
      File "/var/folders/ql/m5k56v8n5sz5880n5sksmc9w0000gn/T/ipykernel_7049/617359052.py", line 67, in <module>
        raise RuntimeError
    RuntimeError
     21%|██████████████████▊                                                                     | 19/89 [01:09<04:28,  3.84s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/5 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     22%|███████████████████▊                                                                    | 20/89 [01:11<03:49,  3.33s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/27 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     24%|████████████████████▊                                                                   | 21/89 [01:15<04:01,  3.56s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/2 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     25%|█████████████████████▊                                                                  | 22/89 [01:17<03:19,  2.98s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/3 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     26%|██████████████████████▋                                                                 | 23/89 [01:21<03:36,  3.28s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/2 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     27%|███████████████████████▋                                                                | 24/89 [01:23<03:18,  3.05s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/2 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     28%|████████████████████████▋                                                               | 25/89 [01:24<02:34,  2.42s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/3 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     29%|█████████████████████████▋                                                              | 26/89 [01:26<02:23,  2.28s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     30%|██████████████████████████▋                                                             | 27/89 [01:28<02:06,  2.05s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/3 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     31%|███████████████████████████▋                                                            | 28/89 [01:29<02:03,  2.02s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     33%|████████████████████████████▋                                                           | 29/89 [01:31<01:47,  1.78s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/3 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     34%|█████████████████████████████▋                                                          | 30/89 [01:32<01:40,  1.70s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/2 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     35%|██████████████████████████████▋                                                         | 31/89 [01:34<01:35,  1.65s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/2 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     36%|███████████████████████████████▋                                                        | 32/89 [01:35<01:30,  1.59s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/2 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     37%|████████████████████████████████▋                                                       | 33/89 [01:39<02:12,  2.37s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/4 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     38%|█████████████████████████████████▌                                                      | 34/89 [01:41<01:52,  2.05s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/2 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     39%|██████████████████████████████████▌                                                     | 35/89 [01:43<01:53,  2.10s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/2 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     40%|███████████████████████████████████▌                                                    | 36/89 [01:44<01:36,  1.83s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/2 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     42%|████████████████████████████████████▌                                                   | 37/89 [01:45<01:26,  1.66s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/2 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     43%|█████████████████████████████████████▌                                                  | 38/89 [01:49<01:52,  2.21s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/2 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     44%|██████████████████████████████████████▌                                                 | 39/89 [01:51<01:48,  2.18s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/2 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     45%|███████████████████████████████████████▌                                                | 40/89 [01:52<01:29,  1.83s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/5 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     46%|████████████████████████████████████████▌                                               | 41/89 [01:53<01:13,  1.53s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/10 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     47%|█████████████████████████████████████████▌                                              | 42/89 [01:54<01:06,  1.42s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/10 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     48%|██████████████████████████████████████████▌                                             | 43/89 [01:55<00:55,  1.21s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/6 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     49%|███████████████████████████████████████████▌                                            | 44/89 [01:56<00:51,  1.15s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/8 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     51%|████████████████████████████████████████████▍                                           | 45/89 [01:57<00:51,  1.18s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/8 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     52%|█████████████████████████████████████████████▍                                          | 46/89 [01:58<00:43,  1.00s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/4 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     53%|██████████████████████████████████████████████▍                                         | 47/89 [01:59<00:45,  1.08s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/5 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     54%|███████████████████████████████████████████████▍                                        | 48/89 [02:00<00:41,  1.01s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/8 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     55%|████████████████████████████████████████████████▍                                       | 49/89 [02:01<00:41,  1.03s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/3 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     56%|█████████████████████████████████████████████████▍                                      | 50/89 [02:02<00:44,  1.14s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/4 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     57%|██████████████████████████████████████████████████▍                                     | 51/89 [02:03<00:39,  1.04s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/7 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     58%|███████████████████████████████████████████████████▍                                    | 52/89 [02:03<00:31,  1.17it/s]


    Retrieving MaterialsDoc documents:   0%|          | 0/3 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     60%|████████████████████████████████████████████████████▍                                   | 53/89 [02:04<00:30,  1.17it/s]


    Retrieving MaterialsDoc documents:   0%|          | 0/5 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     61%|█████████████████████████████████████████████████████▍                                  | 54/89 [02:05<00:29,  1.18it/s]


    Retrieving MaterialsDoc documents:   0%|          | 0/10 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     62%|██████████████████████████████████████████████████████▍                                 | 55/89 [02:07<00:42,  1.24s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/4 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     63%|███████████████████████████████████████████████████████▎                                | 56/89 [02:16<01:59,  3.62s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/8 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     64%|████████████████████████████████████████████████████████▎                               | 57/89 [02:19<01:49,  3.42s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/4 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     65%|█████████████████████████████████████████████████████████▎                              | 58/89 [02:21<01:26,  2.79s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/7 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     66%|██████████████████████████████████████████████████████████▎                             | 59/89 [02:24<01:28,  2.96s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/8 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     67%|███████████████████████████████████████████████████████████▎                            | 60/89 [02:27<01:24,  2.92s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/8 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     69%|████████████████████████████████████████████████████████████▎                           | 61/89 [02:31<01:33,  3.34s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     70%|█████████████████████████████████████████████████████████████▎                          | 62/89 [02:33<01:20,  2.97s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/4 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     71%|██████████████████████████████████████████████████████████████▎                         | 63/89 [02:35<01:09,  2.66s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/5 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     72%|███████████████████████████████████████████████████████████████▎                        | 64/89 [02:38<01:08,  2.74s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/2 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     73%|████████████████████████████████████████████████████████████████▎                       | 65/89 [02:41<01:07,  2.80s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/4 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     74%|█████████████████████████████████████████████████████████████████▎                      | 66/89 [02:43<01:01,  2.67s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/3 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     75%|██████████████████████████████████████████████████████████████████▏                     | 67/89 [02:45<00:51,  2.35s/it]Traceback (most recent call last):
      File "/var/folders/ql/m5k56v8n5sz5880n5sksmc9w0000gn/T/ipykernel_7049/617359052.py", line 54, in <module>
        mids = mpr.get_material_ids(s.composition.reduced_formula)
      File "/Users/shyue/miniconda3/envs/mavrl/lib/python3.9/site-packages/mp_api/client/mprester.py", line 417, in get_material_ids
        for doc in self.materials.search(
      File "/Users/shyue/miniconda3/envs/mavrl/lib/python3.9/site-packages/mp_api/client/routes/materials.py", line 172, in search
        return super()._search(
      File "/Users/shyue/miniconda3/envs/mavrl/lib/python3.9/site-packages/mp_api/client/core/client.py", line 940, in _search
        return self._get_all_documents(
      File "/Users/shyue/miniconda3/envs/mavrl/lib/python3.9/site-packages/mp_api/client/core/client.py", line 987, in _get_all_documents
        results = self._query_resource(
      File "/Users/shyue/miniconda3/envs/mavrl/lib/python3.9/site-packages/mp_api/client/core/client.py", line 288, in _query_resource
        data = self._submit_requests(
      File "/Users/shyue/miniconda3/envs/mavrl/lib/python3.9/site-packages/mp_api/client/core/client.py", line 429, in _submit_requests
        initial_data_tuples = self._multi_thread(
      File "/Users/shyue/miniconda3/envs/mavrl/lib/python3.9/site-packages/mp_api/client/core/client.py", line 643, in _multi_thread
        data, subtotal = future.result()
      File "/Users/shyue/miniconda3/envs/mavrl/lib/python3.9/concurrent/futures/_base.py", line 439, in result
        return self.__get_result()
      File "/Users/shyue/miniconda3/envs/mavrl/lib/python3.9/concurrent/futures/_base.py", line 391, in __get_result
        raise self._exception
      File "/Users/shyue/miniconda3/envs/mavrl/lib/python3.9/concurrent/futures/thread.py", line 58, in run
        result = self.fn(*self.args, **self.kwargs)
      File "/Users/shyue/miniconda3/envs/mavrl/lib/python3.9/site-packages/mp_api/client/core/client.py", line 745, in _submit_request_and_process
        raise MPRestError(
    mp_api.client.core.client.MPRestError: REST query returned with error status code 504 on URL https://api.materialsproject.org/materials/core/?deprecated=False&_fields=material_id&formula=KTaO3&_limit=1000 with message:
    Server timed out trying to obtain data. Try again with a smaller request.
     76%|███████████████████████████████████████████████████████████████████▏                    | 68/89 [03:05<02:42,  7.76s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/4 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     78%|████████████████████████████████████████████████████████████████████▏                   | 69/89 [03:25<03:45, 11.26s/it]Traceback (most recent call last):
      File "/var/folders/ql/m5k56v8n5sz5880n5sksmc9w0000gn/T/ipykernel_7049/617359052.py", line 54, in <module>
        mids = mpr.get_material_ids(s.composition.reduced_formula)
      File "/Users/shyue/miniconda3/envs/mavrl/lib/python3.9/site-packages/mp_api/client/mprester.py", line 417, in get_material_ids
        for doc in self.materials.search(
      File "/Users/shyue/miniconda3/envs/mavrl/lib/python3.9/site-packages/mp_api/client/routes/materials.py", line 172, in search
        return super()._search(
      File "/Users/shyue/miniconda3/envs/mavrl/lib/python3.9/site-packages/mp_api/client/core/client.py", line 940, in _search
        return self._get_all_documents(
      File "/Users/shyue/miniconda3/envs/mavrl/lib/python3.9/site-packages/mp_api/client/core/client.py", line 987, in _get_all_documents
        results = self._query_resource(
      File "/Users/shyue/miniconda3/envs/mavrl/lib/python3.9/site-packages/mp_api/client/core/client.py", line 288, in _query_resource
        data = self._submit_requests(
      File "/Users/shyue/miniconda3/envs/mavrl/lib/python3.9/site-packages/mp_api/client/core/client.py", line 429, in _submit_requests
        initial_data_tuples = self._multi_thread(
      File "/Users/shyue/miniconda3/envs/mavrl/lib/python3.9/site-packages/mp_api/client/core/client.py", line 643, in _multi_thread
        data, subtotal = future.result()
      File "/Users/shyue/miniconda3/envs/mavrl/lib/python3.9/concurrent/futures/_base.py", line 439, in result
        return self.__get_result()
      File "/Users/shyue/miniconda3/envs/mavrl/lib/python3.9/concurrent/futures/_base.py", line 391, in __get_result
        raise self._exception
      File "/Users/shyue/miniconda3/envs/mavrl/lib/python3.9/concurrent/futures/thread.py", line 58, in run
        result = self.fn(*self.args, **self.kwargs)
      File "/Users/shyue/miniconda3/envs/mavrl/lib/python3.9/site-packages/mp_api/client/core/client.py", line 745, in _submit_request_and_process
        raise MPRestError(
    mp_api.client.core.client.MPRestError: REST query returned with error status code 504 on URL https://api.materialsproject.org/materials/core/?deprecated=False&_fields=material_id&formula=EuTiO3&_limit=1000 with message:
    Server timed out trying to obtain data. Try again with a smaller request.
     79%|█████████████████████████████████████████████████████████████████████▏                  | 70/89 [03:46<04:32, 14.35s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     80%|██████████████████████████████████████████████████████████████████████▏                 | 71/89 [04:01<04:20, 14.45s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/3 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     81%|███████████████████████████████████████████████████████████████████████▏                | 72/89 [04:18<04:17, 15.17s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     82%|████████████████████████████████████████████████████████████████████████▏               | 73/89 [04:18<02:52, 10.79s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/6 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     83%|█████████████████████████████████████████████████████████████████████████▏              | 74/89 [04:19<01:55,  7.71s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/2 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     84%|██████████████████████████████████████████████████████████████████████████▏             | 75/89 [04:20<01:18,  5.61s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/5 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     85%|███████████████████████████████████████████████████████████████████████████▏            | 76/89 [04:21<00:54,  4.23s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/12 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     87%|████████████████████████████████████████████████████████████████████████████▏           | 77/89 [04:21<00:37,  3.14s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/4 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     88%|█████████████████████████████████████████████████████████████████████████████           | 78/89 [04:23<00:30,  2.76s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/11 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     89%|██████████████████████████████████████████████████████████████████████████████          | 79/89 [04:24<00:21,  2.13s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/5 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     90%|███████████████████████████████████████████████████████████████████████████████         | 80/89 [04:26<00:18,  2.00s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/4 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     91%|████████████████████████████████████████████████████████████████████████████████        | 81/89 [04:26<00:12,  1.58s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/2 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     92%|█████████████████████████████████████████████████████████████████████████████████       | 82/89 [04:27<00:09,  1.33s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/9 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     93%|██████████████████████████████████████████████████████████████████████████████████      | 83/89 [04:28<00:07,  1.23s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/14 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     94%|███████████████████████████████████████████████████████████████████████████████████     | 84/89 [04:29<00:06,  1.22s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/15 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     96%|████████████████████████████████████████████████████████████████████████████████████    | 85/89 [04:30<00:04,  1.01s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/20 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     97%|█████████████████████████████████████████████████████████████████████████████████████   | 86/89 [04:30<00:02,  1.20it/s]


    Retrieving MaterialsDoc documents:   0%|          | 0/11 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


    Traceback (most recent call last):
      File "/var/folders/ql/m5k56v8n5sz5880n5sksmc9w0000gn/T/ipykernel_7049/617359052.py", line 67, in <module>
        raise RuntimeError
    RuntimeError
     98%|██████████████████████████████████████████████████████████████████████████████████████  | 87/89 [04:32<00:02,  1.22s/it]


    Retrieving MaterialsDoc documents:   0%|          | 0/6 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


     99%|███████████████████████████████████████████████████████████████████████████████████████ | 88/89 [04:32<00:00,  1.05it/s]


    Retrieving MaterialsDoc documents:   0%|          | 0/6 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]



    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


    100%|████████████████████████████████████████████████████████████████████████████████████████| 89/89 [04:34<00:00,  3.09s/it]



```python
data["% error vs Expt"] = (data["Predicted a (Å)"] - data["a (Å)"]) / data["a (Å)"]
data["% error vs MP"] = (data["Predicted a (Å)"] - data["MP a (Å)"]) / data["MP a (Å)"]
```


```python
data.sort_index().style.format({"% error vs Expt": "{:,.2%}", "% error vs MP": "{:,.2%}"}).background_gradient()
```




<style type="text/css">
#T_1b734_row0_col1 {
  background-color: #76aad0;
  color: #f1f1f1;
}
#T_1b734_row0_col2 {
  background-color: #056caa;
  color: #f1f1f1;
}
#T_1b734_row0_col3 {
  background-color: #4697c4;
  color: #f1f1f1;
}
#T_1b734_row0_col4, #T_1b734_row26_col1, #T_1b734_row68_col2, #T_1b734_row68_col3 {
  background-color: #023858;
  color: #f1f1f1;
}
#T_1b734_row0_col5, #T_1b734_row1_col5, #T_1b734_row2_col5, #T_1b734_row3_col5, #T_1b734_row4_col5, #T_1b734_row5_col5, #T_1b734_row6_col5, #T_1b734_row7_col5, #T_1b734_row8_col5, #T_1b734_row9_col5, #T_1b734_row10_col5, #T_1b734_row11_col5, #T_1b734_row12_col5, #T_1b734_row13_col5, #T_1b734_row14_col5, #T_1b734_row15_col5, #T_1b734_row16_col5, #T_1b734_row17_col5, #T_1b734_row18_col5, #T_1b734_row19_col5, #T_1b734_row20_col5, #T_1b734_row21_col5, #T_1b734_row22_col5, #T_1b734_row23_col5, #T_1b734_row24_col5, #T_1b734_row25_col5, #T_1b734_row26_col5, #T_1b734_row27_col5, #T_1b734_row28_col5, #T_1b734_row29_col5, #T_1b734_row30_col5, #T_1b734_row31_col5, #T_1b734_row32_col5, #T_1b734_row33_col5, #T_1b734_row34_col5, #T_1b734_row35_col5, #T_1b734_row36_col5, #T_1b734_row37_col5, #T_1b734_row38_col5, #T_1b734_row39_col5, #T_1b734_row40_col5, #T_1b734_row41_col5, #T_1b734_row42_col5, #T_1b734_row43_col5, #T_1b734_row44_col5, #T_1b734_row45_col5, #T_1b734_row46_col5, #T_1b734_row47_col5, #T_1b734_row48_col5, #T_1b734_row49_col5, #T_1b734_row50_col5, #T_1b734_row51_col5, #T_1b734_row52_col5, #T_1b734_row53_col5, #T_1b734_row54_col5, #T_1b734_row55_col5, #T_1b734_row56_col5, #T_1b734_row57_col5, #T_1b734_row58_col5, #T_1b734_row59_col5, #T_1b734_row60_col5, #T_1b734_row61_col5, #T_1b734_row62_col5, #T_1b734_row63_col5, #T_1b734_row64_col5, #T_1b734_row65_col5, #T_1b734_row66_col5, #T_1b734_row67_col5, #T_1b734_row68_col5, #T_1b734_row69_col5, #T_1b734_row70_col5, #T_1b734_row71_col5, #T_1b734_row72_col5, #T_1b734_row73_col5, #T_1b734_row74_col5, #T_1b734_row75_col5, #T_1b734_row76_col5, #T_1b734_row77_col5, #T_1b734_row78_col5, #T_1b734_row79_col5, #T_1b734_row80_col5, #T_1b734_row81_col5, #T_1b734_row82_col5, #T_1b734_row83_col5, #T_1b734_row84_col5, #T_1b734_row85_col5, #T_1b734_row86_col5, #T_1b734_row87_col5, #T_1b734_row88_col5 {
  background-color: #000000;
  color: #f1f1f1;
}
#T_1b734_row1_col1 {
  background-color: #d1d2e6;
  color: #000000;
}
#T_1b734_row1_col2, #T_1b734_row86_col3 {
  background-color: #589ec8;
  color: #f1f1f1;
}
#T_1b734_row1_col3, #T_1b734_row7_col3 {
  background-color: #c5cce3;
  color: #000000;
}
#T_1b734_row1_col4, #T_1b734_row16_col4, #T_1b734_row23_col4, #T_1b734_row34_col4, #T_1b734_row68_col4 {
  background-color: #034871;
  color: #f1f1f1;
}
#T_1b734_row2_col1 {
  background-color: #d2d3e7;
  color: #000000;
}
#T_1b734_row2_col2, #T_1b734_row12_col1 {
  background-color: #5c9fc9;
  color: #f1f1f1;
}
#T_1b734_row2_col3, #T_1b734_row19_col1, #T_1b734_row82_col1 {
  background-color: #cdd0e5;
  color: #000000;
}
#T_1b734_row2_col4, #T_1b734_row8_col4, #T_1b734_row46_col4, #T_1b734_row51_col4, #T_1b734_row66_col2, #T_1b734_row70_col4, #T_1b734_row79_col4, #T_1b734_row82_col4, #T_1b734_row84_col4 {
  background-color: #045483;
  color: #f1f1f1;
}
#T_1b734_row3_col1, #T_1b734_row4_col3, #T_1b734_row7_col2, #T_1b734_row28_col1, #T_1b734_row29_col3, #T_1b734_row31_col1, #T_1b734_row67_col1, #T_1b734_row81_col2, #T_1b734_row84_col3 {
  background-color: #549cc7;
  color: #f1f1f1;
}
#T_1b734_row3_col2, #T_1b734_row31_col2 {
  background-color: #056dab;
  color: #f1f1f1;
}
#T_1b734_row3_col3, #T_1b734_row9_col2, #T_1b734_row67_col3 {
  background-color: #3b92c1;
  color: #f1f1f1;
}
#T_1b734_row3_col4, #T_1b734_row42_col4, #T_1b734_row49_col4, #T_1b734_row61_col4, #T_1b734_row67_col4, #T_1b734_row69_col4 {
  background-color: #034d79;
  color: #f1f1f1;
}
#T_1b734_row4_col1, #T_1b734_row29_col1 {
  background-color: #67a4cc;
  color: #f1f1f1;
}
#T_1b734_row4_col2, #T_1b734_row17_col4, #T_1b734_row29_col2 {
  background-color: #0872b1;
  color: #f1f1f1;
}
#T_1b734_row4_col4, #T_1b734_row11_col4, #T_1b734_row24_col4, #T_1b734_row29_col4, #T_1b734_row48_col4, #T_1b734_row55_col4, #T_1b734_row68_col1, #T_1b734_row75_col4, #T_1b734_row83_col4, #T_1b734_row86_col4 {
  background-color: #03517e;
  color: #f1f1f1;
}
#T_1b734_row5_col1 {
  background-color: #2786bb;
  color: #f1f1f1;
}
#T_1b734_row5_col2 {
  background-color: #046097;
  color: #f1f1f1;
}
#T_1b734_row5_col3, #T_1b734_row6_col2, #T_1b734_row30_col3 {
  background-color: #0f76b3;
  color: #f1f1f1;
}
#T_1b734_row5_col4, #T_1b734_row14_col4, #T_1b734_row35_col4, #T_1b734_row39_col4, #T_1b734_row40_col4, #T_1b734_row41_col4, #T_1b734_row42_col3, #T_1b734_row47_col4, #T_1b734_row52_col4, #T_1b734_row57_col4, #T_1b734_row62_col4 {
  background-color: #034c78;
  color: #f1f1f1;
}
#T_1b734_row6_col1, #T_1b734_row24_col2 {
  background-color: #79abd0;
  color: #f1f1f1;
}
#T_1b734_row6_col3, #T_1b734_row19_col3, #T_1b734_row51_col1, #T_1b734_row79_col1, #T_1b734_row81_col3 {
  background-color: #c6cce3;
  color: #000000;
}
#T_1b734_row6_col4, #T_1b734_row69_col2 {
  background-color: #6da6cd;
  color: #f1f1f1;
}
#T_1b734_row7_col1, #T_1b734_row64_col3 {
  background-color: #d2d2e7;
  color: #000000;
}
#T_1b734_row7_col4, #T_1b734_row15_col4, #T_1b734_row36_col4 {
  background-color: #03476f;
  color: #f1f1f1;
}
#T_1b734_row8_col1 {
  background-color: #e6e2ef;
  color: #000000;
}
#T_1b734_row8_col2, #T_1b734_row46_col3 {
  background-color: #78abd0;
  color: #f1f1f1;
}
#T_1b734_row8_col3, #T_1b734_row24_col3 {
  background-color: #e3e0ee;
  color: #000000;
}
#T_1b734_row9_col1 {
  background-color: #b3c3de;
  color: #000000;
}
#T_1b734_row9_col3 {
  background-color: #a9bfdc;
  color: #000000;
}
#T_1b734_row9_col4, #T_1b734_row32_col4, #T_1b734_row36_col2, #T_1b734_row59_col4, #T_1b734_row78_col4 {
  background-color: #045382;
  color: #f1f1f1;
}
#T_1b734_row10_col1 {
  background-color: #8eb3d5;
  color: #000000;
}
#T_1b734_row10_col2, #T_1b734_row26_col2, #T_1b734_row26_col4, #T_1b734_row27_col1, #T_1b734_row27_col3, #T_1b734_row43_col2, #T_1b734_row61_col2 {
  background-color: #fff7fb;
  color: #000000;
}
#T_1b734_row10_col3 {
  background-color: #94b6d7;
  color: #000000;
}
#T_1b734_row10_col4, #T_1b734_row39_col3, #T_1b734_row72_col2 {
  background-color: #04639b;
  color: #f1f1f1;
}
#T_1b734_row11_col1, #T_1b734_row59_col3 {
  background-color: #e8e4f0;
  color: #000000;
}
#T_1b734_row11_col2 {
  background-color: #7bacd1;
  color: #f1f1f1;
}
#T_1b734_row11_col3 {
  background-color: #e4e1ef;
  color: #000000;
}
#T_1b734_row12_col2, #T_1b734_row39_col1, #T_1b734_row40_col3, #T_1b734_row53_col2 {
  background-color: #056fae;
  color: #f1f1f1;
}
#T_1b734_row12_col3 {
  background-color: #509ac6;
  color: #f1f1f1;
}
#T_1b734_row12_col4, #T_1b734_row73_col4 {
  background-color: #04598c;
  color: #f1f1f1;
}
#T_1b734_row13_col1 {
  background-color: #dfddec;
  color: #000000;
}
#T_1b734_row13_col2, #T_1b734_row86_col1 {
  background-color: #6ba5cd;
  color: #f1f1f1;
}
#T_1b734_row13_col3, #T_1b734_row64_col1 {
  background-color: #d9d8ea;
  color: #000000;
}
#T_1b734_row13_col4, #T_1b734_row22_col4, #T_1b734_row31_col4, #T_1b734_row54_col4, #T_1b734_row66_col4, #T_1b734_row74_col4 {
  background-color: #034a74;
  color: #f1f1f1;
}
#T_1b734_row14_col1 {
  background-color: #4295c3;
  color: #f1f1f1;
}
#T_1b734_row14_col2, #T_1b734_row35_col2, #T_1b734_row85_col4 {
  background-color: #0567a2;
  color: #f1f1f1;
}
#T_1b734_row14_col3 {
  background-color: #2987bc;
  color: #f1f1f1;
}
#T_1b734_row15_col1, #T_1b734_row20_col1, #T_1b734_row34_col1 {
  background-color: #2d8abd;
  color: #f1f1f1;
}
#T_1b734_row15_col2, #T_1b734_row30_col2, #T_1b734_row66_col3 {
  background-color: #046299;
  color: #f1f1f1;
}
#T_1b734_row15_col3, #T_1b734_row41_col2 {
  background-color: #1077b4;
  color: #f1f1f1;
}
#T_1b734_row16_col1, #T_1b734_row36_col1, #T_1b734_row38_col2 {
  background-color: #0c74b2;
  color: #f1f1f1;
}
#T_1b734_row16_col2, #T_1b734_row65_col3, #T_1b734_row72_col4, #T_1b734_row77_col4, #T_1b734_row81_col4 {
  background-color: #045687;
  color: #f1f1f1;
}
#T_1b734_row16_col3 {
  background-color: #05659f;
  color: #f1f1f1;
}
#T_1b734_row17_col1, #T_1b734_row44_col2, #T_1b734_row60_col3 {
  background-color: #81aed2;
  color: #f1f1f1;
}
#T_1b734_row17_col2, #T_1b734_row48_col1 {
  background-color: #328dbf;
  color: #f1f1f1;
}
#T_1b734_row17_col3, #T_1b734_row18_col2, #T_1b734_row54_col3, #T_1b734_row87_col3 {
  background-color: #9cb9d9;
  color: #000000;
}
#T_1b734_row18_col1, #T_1b734_row18_col3 {
  background-color: #fef6fb;
  color: #000000;
}
#T_1b734_row18_col4, #T_1b734_row27_col4, #T_1b734_row62_col2 {
  background-color: #045788;
  color: #f1f1f1;
}
#T_1b734_row19_col2, #T_1b734_row49_col2, #T_1b734_row51_col2, #T_1b734_row58_col2 {
  background-color: #529bc7;
  color: #f1f1f1;
}
#T_1b734_row19_col4, #T_1b734_row39_col2 {
  background-color: #045585;
  color: #f1f1f1;
}
#T_1b734_row20_col2 {
  background-color: #045f95;
  color: #f1f1f1;
}
#T_1b734_row20_col3, #T_1b734_row63_col2, #T_1b734_row73_col2, #T_1b734_row74_col2 {
  background-color: #65a3cb;
  color: #f1f1f1;
}
#T_1b734_row20_col4, #T_1b734_row60_col2, #T_1b734_row61_col3 {
  background-color: #2182b9;
  color: #f1f1f1;
}
#T_1b734_row21_col1, #T_1b734_row43_col3 {
  background-color: #ced0e6;
  color: #000000;
}
#T_1b734_row21_col2, #T_1b734_row53_col1, #T_1b734_row71_col3, #T_1b734_row82_col2 {
  background-color: #569dc8;
  color: #f1f1f1;
}
#T_1b734_row21_col3 {
  background-color: #c2cbe2;
  color: #000000;
}
#T_1b734_row21_col4, #T_1b734_row30_col4, #T_1b734_row60_col4, #T_1b734_row63_col4, #T_1b734_row65_col2 {
  background-color: #034973;
  color: #f1f1f1;
}
#T_1b734_row22_col1, #T_1b734_row87_col2 {
  background-color: #308cbe;
  color: #f1f1f1;
}
#T_1b734_row22_col2, #T_1b734_row36_col3, #T_1b734_row65_col1 {
  background-color: #04649e;
  color: #f1f1f1;
}
#T_1b734_row22_col3 {
  background-color: #167bb6;
  color: #f1f1f1;
}
#T_1b734_row23_col1, #T_1b734_row57_col3, #T_1b734_row88_col1 {
  background-color: #b0c2de;
  color: #000000;
}
#T_1b734_row23_col2 {
  background-color: #348ebf;
  color: #f1f1f1;
}
#T_1b734_row23_col3 {
  background-color: #9fbad9;
  color: #000000;
}
#T_1b734_row24_col1 {
  background-color: #e7e3f0;
  color: #000000;
}
#T_1b734_row25_col1, #T_1b734_row25_col3 {
  background-color: #adc1dd;
  color: #000000;
}
#T_1b734_row25_col2, #T_1b734_row35_col1 {
  background-color: #3f93c2;
  color: #f1f1f1;
}
#T_1b734_row25_col4 {
  background-color: #045e93;
  color: #f1f1f1;
}
#T_1b734_row26_col3 {
  background-color: #d4d4e8;
  color: #000000;
}
#T_1b734_row27_col2 {
  background-color: #a1bbda;
  color: #000000;
}
#T_1b734_row28_col2 {
  background-color: #056ba7;
  color: #f1f1f1;
}
#T_1b734_row28_col3, #T_1b734_row54_col2 {
  background-color: #3991c1;
  color: #f1f1f1;
}
#T_1b734_row28_col4, #T_1b734_row64_col4, #T_1b734_row65_col4 {
  background-color: #034b76;
  color: #f1f1f1;
}
#T_1b734_row30_col1 {
  background-color: #2a88bc;
  color: #f1f1f1;
}
#T_1b734_row31_col3, #T_1b734_row32_col2, #T_1b734_row52_col1 {
  background-color: #358fc0;
  color: #f1f1f1;
}
#T_1b734_row32_col1, #T_1b734_row33_col3 {
  background-color: #abbfdc;
  color: #000000;
}
#T_1b734_row32_col3 {
  background-color: #a2bcda;
  color: #000000;
}
#T_1b734_row33_col1, #T_1b734_row57_col1 {
  background-color: #bcc7e1;
  color: #000000;
}
#T_1b734_row33_col2, #T_1b734_row53_col3, #T_1b734_row70_col2 {
  background-color: #3d93c2;
  color: #f1f1f1;
}
#T_1b734_row33_col4 {
  background-color: #03446a;
  color: #f1f1f1;
}
#T_1b734_row34_col2 {
  background-color: #04629a;
  color: #f1f1f1;
}
#T_1b734_row34_col3 {
  background-color: #1278b4;
  color: #f1f1f1;
}
#T_1b734_row35_col3 {
  background-color: #2685bb;
  color: #f1f1f1;
}
#T_1b734_row37_col1, #T_1b734_row74_col1 {
  background-color: #dcdaeb;
  color: #000000;
}
#T_1b734_row37_col2, #T_1b734_row71_col1 {
  background-color: #69a5cc;
  color: #f1f1f1;
}
#T_1b734_row37_col3 {
  background-color: #d7d6e9;
  color: #000000;
}
#T_1b734_row37_col4, #T_1b734_row45_col4, #T_1b734_row50_col4, #T_1b734_row88_col4 {
  background-color: #034f7d;
  color: #f1f1f1;
}
#T_1b734_row38_col1 {
  background-color: #7dacd1;
  color: #f1f1f1;
}
#T_1b734_row38_col3 {
  background-color: #60a1ca;
  color: #f1f1f1;
}
#T_1b734_row38_col4 {
  background-color: #03466e;
  color: #f1f1f1;
}
#T_1b734_row40_col1 {
  background-color: #1b7eb7;
  color: #f1f1f1;
}
#T_1b734_row40_col2, #T_1b734_row42_col1 {
  background-color: #045e94;
  color: #f1f1f1;
}
#T_1b734_row41_col1 {
  background-color: #73a9cf;
  color: #f1f1f1;
}
#T_1b734_row41_col3, #T_1b734_row47_col2 {
  background-color: #5a9ec9;
  color: #f1f1f1;
}
#T_1b734_row42_col2 {
  background-color: #034369;
  color: #f1f1f1;
}
#T_1b734_row43_col1, #T_1b734_row73_col1, #T_1b734_row74_col3 {
  background-color: #d5d5e8;
  color: #000000;
}
#T_1b734_row43_col4, #T_1b734_row53_col4 {
  background-color: #034e7b;
  color: #f1f1f1;
}
#T_1b734_row44_col1 {
  background-color: #ece7f2;
  color: #000000;
}
#T_1b734_row44_col3, #T_1b734_row59_col1 {
  background-color: #ebe6f2;
  color: #000000;
}
#T_1b734_row44_col4, #T_1b734_row55_col2 {
  background-color: #045a8d;
  color: #f1f1f1;
}
#T_1b734_row45_col1, #T_1b734_row64_col2, #T_1b734_row84_col1 {
  background-color: #63a2cb;
  color: #f1f1f1;
}
#T_1b734_row45_col2, #T_1b734_row71_col2 {
  background-color: #0a73b2;
  color: #f1f1f1;
}
#T_1b734_row45_col3, #T_1b734_row79_col2 {
  background-color: #4e9ac6;
  color: #f1f1f1;
}
#T_1b734_row46_col1 {
  background-color: #83afd3;
  color: #f1f1f1;
}
#T_1b734_row46_col2 {
  background-color: #1c7fb8;
  color: #f1f1f1;
}
#T_1b734_row47_col1, #T_1b734_row63_col3, #T_1b734_row73_col3 {
  background-color: #d3d4e7;
  color: #000000;
}
#T_1b734_row47_col3 {
  background-color: #cacee5;
  color: #000000;
}
#T_1b734_row48_col2 {
  background-color: #0566a0;
  color: #f1f1f1;
}
#T_1b734_row48_col3, #T_1b734_row52_col3, #T_1b734_row72_col3, #T_1b734_row77_col2 {
  background-color: #1e80b8;
  color: #f1f1f1;
}
#T_1b734_row49_col1, #T_1b734_row82_col3 {
  background-color: #c8cde4;
  color: #000000;
}
#T_1b734_row49_col3 {
  background-color: #bfc9e1;
  color: #000000;
}
#T_1b734_row50_col1 {
  background-color: #f7f0f7;
  color: #000000;
}
#T_1b734_row50_col2, #T_1b734_row83_col2 {
  background-color: #91b5d6;
  color: #000000;
}
#T_1b734_row50_col3 {
  background-color: #f5eef6;
  color: #000000;
}
#T_1b734_row51_col3 {
  background-color: #c1cae2;
  color: #000000;
}
#T_1b734_row52_col2 {
  background-color: #0567a1;
  color: #f1f1f1;
}
#T_1b734_row54_col1, #T_1b734_row70_col3 {
  background-color: #acc0dd;
  color: #000000;
}
#T_1b734_row55_col1, #T_1b734_row62_col1, #T_1b734_row84_col2, #T_1b734_row86_col2 {
  background-color: #0d75b3;
  color: #f1f1f1;
}
#T_1b734_row55_col3 {
  background-color: #056aa6;
  color: #f1f1f1;
}
#T_1b734_row56_col1 {
  background-color: #f2ecf5;
  color: #000000;
}
#T_1b734_row56_col2 {
  background-color: #88b1d4;
  color: #000000;
}
#T_1b734_row56_col3, #T_1b734_row75_col3 {
  background-color: #f0eaf4;
  color: #000000;
}
#T_1b734_row56_col4, #T_1b734_row71_col4, #T_1b734_row76_col4, #T_1b734_row87_col4 {
  background-color: #045280;
  color: #f1f1f1;
}
#T_1b734_row57_col2, #T_1b734_row76_col2 {
  background-color: #4094c3;
  color: #f1f1f1;
}
#T_1b734_row58_col1, #T_1b734_row58_col3, #T_1b734_row78_col3, #T_1b734_row85_col3 {
  background-color: #b9c6e0;
  color: #000000;
}
#T_1b734_row58_col4 {
  background-color: #045c90;
  color: #f1f1f1;
}
#T_1b734_row59_col2 {
  background-color: #80aed2;
  color: #f1f1f1;
}
#T_1b734_row60_col1 {
  background-color: #96b6d7;
  color: #000000;
}
#T_1b734_row61_col1, #T_1b734_row88_col2 {
  background-color: #3790c0;
  color: #f1f1f1;
}
#T_1b734_row62_col3 {
  background-color: #0568a3;
  color: #f1f1f1;
}
#T_1b734_row63_col1 {
  background-color: #dbdaeb;
  color: #000000;
}
#T_1b734_row66_col1 {
  background-color: #056faf;
  color: #f1f1f1;
}
#T_1b734_row67_col2 {
  background-color: #056dac;
  color: #f1f1f1;
}
#T_1b734_row69_col1 {
  background-color: #dedcec;
  color: #000000;
}
#T_1b734_row69_col3 {
  background-color: #d8d7e9;
  color: #000000;
}
#T_1b734_row70_col1 {
  background-color: #b4c4df;
  color: #000000;
}
#T_1b734_row72_col1 {
  background-color: #2c89bd;
  color: #f1f1f1;
}
#T_1b734_row75_col1 {
  background-color: #f1ebf5;
  color: #000000;
}
#T_1b734_row75_col2, #T_1b734_row77_col1 {
  background-color: #89b1d4;
  color: #000000;
}
#T_1b734_row76_col1 {
  background-color: #b8c6e0;
  color: #000000;
}
#T_1b734_row76_col3, #T_1b734_row85_col1 {
  background-color: #afc1dd;
  color: #000000;
}
#T_1b734_row77_col3 {
  background-color: #7eadd1;
  color: #f1f1f1;
}
#T_1b734_row78_col1, #T_1b734_row79_col3 {
  background-color: #c0c9e2;
  color: #000000;
}
#T_1b734_row78_col2, #T_1b734_row85_col2 {
  background-color: #4897c4;
  color: #f1f1f1;
}
#T_1b734_row80_col1 {
  background-color: #faf2f8;
  color: #000000;
}
#T_1b734_row80_col2 {
  background-color: #9ab8d8;
  color: #000000;
}
#T_1b734_row80_col3 {
  background-color: #faf3f9;
  color: #000000;
}
#T_1b734_row80_col4 {
  background-color: #045b8f;
  color: #f1f1f1;
}
#T_1b734_row81_col1 {
  background-color: #cccfe5;
  color: #000000;
}
#T_1b734_row83_col1 {
  background-color: #f6eff7;
  color: #000000;
}
#T_1b734_row83_col3 {
  background-color: #f4eef6;
  color: #000000;
}
#T_1b734_row87_col1 {
  background-color: #a7bddb;
  color: #000000;
}
#T_1b734_row88_col3 {
  background-color: #a5bddb;
  color: #000000;
}
</style>
<table id="T_1b734">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_1b734_level0_col0" class="col_heading level0 col0" >Crystal structure</th>
      <th id="T_1b734_level0_col1" class="col_heading level0 col1" >a (Å)</th>
      <th id="T_1b734_level0_col2" class="col_heading level0 col2" >MP a (Å)</th>
      <th id="T_1b734_level0_col3" class="col_heading level0 col3" >Predicted a (Å)</th>
      <th id="T_1b734_level0_col4" class="col_heading level0 col4" >% error vs Expt</th>
      <th id="T_1b734_level0_col5" class="col_heading level0 col5" >% error vs MP</th>
    </tr>
    <tr>
      <th class="index_name level0" >Material</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
      <th class="blank col5" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_1b734_level0_row0" class="row_heading level0 row0" >Ac</th>
      <td id="T_1b734_row0_col0" class="data row0 col0" >FCC</td>
      <td id="T_1b734_row0_col1" class="data row0 col1" >5.310000</td>
      <td id="T_1b734_row0_col2" class="data row0 col2" >5.696211</td>
      <td id="T_1b734_row0_col3" class="data row0 col3" >5.613508</td>
      <td id="T_1b734_row0_col4" class="data row0 col4" >5.72%</td>
      <td id="T_1b734_row0_col5" class="data row0 col5" >-1.45%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row1" class="row_heading level0 row1" >Ag</th>
      <td id="T_1b734_row1_col0" class="data row1 col0" >FCC</td>
      <td id="T_1b734_row1_col1" class="data row1 col1" >4.079000</td>
      <td id="T_1b734_row1_col2" class="data row1 col2" >4.104356</td>
      <td id="T_1b734_row1_col3" class="data row1 col3" >4.172203</td>
      <td id="T_1b734_row1_col4" class="data row1 col4" >2.28%</td>
      <td id="T_1b734_row1_col5" class="data row1 col5" >1.65%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row2" class="row_heading level0 row2" >Al</th>
      <td id="T_1b734_row2_col0" class="data row2 col0" >FCC</td>
      <td id="T_1b734_row2_col1" class="data row2 col1" >4.046000</td>
      <td id="T_1b734_row2_col2" class="data row2 col2" >4.038930</td>
      <td id="T_1b734_row2_col3" class="data row2 col3" >4.046954</td>
      <td id="T_1b734_row2_col4" class="data row2 col4" >0.02%</td>
      <td id="T_1b734_row2_col5" class="data row2 col5" >0.20%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row3" class="row_heading level0 row3" >AlAs</th>
      <td id="T_1b734_row3_col0" class="data row3 col0" >Zinc blende (FCC)</td>
      <td id="T_1b734_row3_col1" class="data row3 col1" >5.660500</td>
      <td id="T_1b734_row3_col2" class="data row3 col2" >5.675802</td>
      <td id="T_1b734_row3_col3" class="data row3 col3" >5.730905</td>
      <td id="T_1b734_row3_col4" class="data row3 col4" >1.24%</td>
      <td id="T_1b734_row3_col5" class="data row3 col5" >0.97%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row4" class="row_heading level0 row4" >AlP</th>
      <td id="T_1b734_row4_col0" class="data row4 col0" >Zinc blende (FCC)</td>
      <td id="T_1b734_row4_col1" class="data row4 col1" >5.451000</td>
      <td id="T_1b734_row4_col2" class="data row4 col2" >5.472967</td>
      <td id="T_1b734_row4_col3" class="data row4 col3" >5.489397</td>
      <td id="T_1b734_row4_col4" class="data row4 col4" >0.70%</td>
      <td id="T_1b734_row4_col5" class="data row4 col5" >0.30%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row5" class="row_heading level0 row5" >AlSb</th>
      <td id="T_1b734_row5_col0" class="data row5 col0" >Zinc blende (FCC)</td>
      <td id="T_1b734_row5_col1" class="data row5 col1" >6.135500</td>
      <td id="T_1b734_row5_col2" class="data row5 col2" >6.185042</td>
      <td id="T_1b734_row5_col3" class="data row5 col3" >6.228938</td>
      <td id="T_1b734_row5_col4" class="data row5 col4" >1.52%</td>
      <td id="T_1b734_row5_col5" class="data row5 col5" >0.71%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row6" class="row_heading level0 row6" >Ar</th>
      <td id="T_1b734_row6_col0" class="data row6 col0" >FCC</td>
      <td id="T_1b734_row6_col1" class="data row6 col1" >5.260000</td>
      <td id="T_1b734_row6_col2" class="data row6 col2" >5.363160</td>
      <td id="T_1b734_row6_col3" class="data row6 col3" >4.145401</td>
      <td id="T_1b734_row6_col4" class="data row6 col4" >-21.19%</td>
      <td id="T_1b734_row6_col5" class="data row6 col5" >-22.71%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row7" class="row_heading level0 row7" >Au</th>
      <td id="T_1b734_row7_col0" class="data row7 col0" >FCC</td>
      <td id="T_1b734_row7_col1" class="data row7 col1" >4.065000</td>
      <td id="T_1b734_row7_col2" class="data row7 col2" >4.171289</td>
      <td id="T_1b734_row7_col3" class="data row7 col3" >4.168613</td>
      <td id="T_1b734_row7_col4" class="data row7 col4" >2.55%</td>
      <td id="T_1b734_row7_col5" class="data row7 col5" >-0.06%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row8" class="row_heading level0 row8" >BN</th>
      <td id="T_1b734_row8_col0" class="data row8 col0" >Zinc blende (FCC)</td>
      <td id="T_1b734_row8_col1" class="data row8 col1" >3.615000</td>
      <td id="T_1b734_row8_col2" class="data row8 col2" >3.626002</td>
      <td id="T_1b734_row8_col3" class="data row8 col3" >3.615134</td>
      <td id="T_1b734_row8_col4" class="data row8 col4" >0.00%</td>
      <td id="T_1b734_row8_col5" class="data row8 col5" >-0.30%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row9" class="row_heading level0 row9" >BP</th>
      <td id="T_1b734_row9_col0" class="data row9 col0" >Zinc blende (FCC)</td>
      <td id="T_1b734_row9_col1" class="data row9 col1" >4.538000</td>
      <td id="T_1b734_row9_col2" class="data row9 col2" >4.532145</td>
      <td id="T_1b734_row9_col3" class="data row9 col3" >4.543347</td>
      <td id="T_1b734_row9_col4" class="data row9 col4" >0.12%</td>
      <td id="T_1b734_row9_col5" class="data row9 col5" >0.25%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row10" class="row_heading level0 row10" >Ba</th>
      <td id="T_1b734_row10_col0" class="data row10 col0" >BCC</td>
      <td id="T_1b734_row10_col1" class="data row10 col1" >5.020000</td>
      <td id="T_1b734_row10_col2" class="data row10 col2" >0.000000</td>
      <td id="T_1b734_row10_col3" class="data row10 col3" >4.814511</td>
      <td id="T_1b734_row10_col4" class="data row10 col4" >-4.09%</td>
      <td id="T_1b734_row10_col5" class="data row10 col5" >inf%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row11" class="row_heading level0 row11" >C (diamond)</th>
      <td id="T_1b734_row11_col0" class="data row11 col0" >Diamond (FCC)</td>
      <td id="T_1b734_row11_col1" class="data row11 col1" >3.567000</td>
      <td id="T_1b734_row11_col2" class="data row11 col2" >3.560745</td>
      <td id="T_1b734_row11_col3" class="data row11 col3" >3.590531</td>
      <td id="T_1b734_row11_col4" class="data row11 col4" >0.66%</td>
      <td id="T_1b734_row11_col5" class="data row11 col5" >0.84%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row12" class="row_heading level0 row12" >Ca</th>
      <td id="T_1b734_row12_col0" class="data row12 col0" >FCC</td>
      <td id="T_1b734_row12_col1" class="data row12 col1" >5.580000</td>
      <td id="T_1b734_row12_col2" class="data row12 col2" >5.576816</td>
      <td id="T_1b734_row12_col3" class="data row12 col3" >5.513572</td>
      <td id="T_1b734_row12_col4" class="data row12 col4" >-1.19%</td>
      <td id="T_1b734_row12_col5" class="data row12 col5" >-1.13%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row13" class="row_heading level0 row13" >CaVO3</th>
      <td id="T_1b734_row13_col0" class="data row13 col0" >Cubic perovskite</td>
      <td id="T_1b734_row13_col1" class="data row13 col1" >3.767000</td>
      <td id="T_1b734_row13_col2" class="data row13 col2" >3.830406</td>
      <td id="T_1b734_row13_col3" class="data row13 col3" >3.840966</td>
      <td id="T_1b734_row13_col4" class="data row13 col4" >1.96%</td>
      <td id="T_1b734_row13_col5" class="data row13 col5" >0.28%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row14" class="row_heading level0 row14" >CdS</th>
      <td id="T_1b734_row14_col0" class="data row14 col0" >Zinc blende (FCC)</td>
      <td id="T_1b734_row14_col1" class="data row14 col1" >5.832000</td>
      <td id="T_1b734_row14_col2" class="data row14 col2" >5.885907</td>
      <td id="T_1b734_row14_col3" class="data row14 col3" >5.922216</td>
      <td id="T_1b734_row14_col4" class="data row14 col4" >1.55%</td>
      <td id="T_1b734_row14_col5" class="data row14 col5" >0.62%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row15" class="row_heading level0 row15" >CdSe</th>
      <td id="T_1b734_row15_col0" class="data row15 col0" >Zinc blende (FCC)</td>
      <td id="T_1b734_row15_col1" class="data row15 col1" >6.050000</td>
      <td id="T_1b734_row15_col2" class="data row15 col2" >6.140542</td>
      <td id="T_1b734_row15_col3" class="data row15 col3" >6.211245</td>
      <td id="T_1b734_row15_col4" class="data row15 col4" >2.67%</td>
      <td id="T_1b734_row15_col5" class="data row15 col5" >1.15%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row16" class="row_heading level0 row16" >CdTe</th>
      <td id="T_1b734_row16_col0" class="data row16 col0" >Zinc blende (FCC)</td>
      <td id="T_1b734_row16_col1" class="data row16 col1" >6.482000</td>
      <td id="T_1b734_row16_col2" class="data row16 col2" >6.564227</td>
      <td id="T_1b734_row16_col3" class="data row16 col3" >6.630309</td>
      <td id="T_1b734_row16_col4" class="data row16 col4" >2.29%</td>
      <td id="T_1b734_row16_col5" class="data row16 col5" >1.01%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row17" class="row_heading level0 row17" >Ce</th>
      <td id="T_1b734_row17_col0" class="data row17 col0" >FCC</td>
      <td id="T_1b734_row17_col1" class="data row17 col1" >5.160000</td>
      <td id="T_1b734_row17_col2" class="data row17 col2" >4.672429</td>
      <td id="T_1b734_row17_col3" class="data row17 col3" >4.712238</td>
      <td id="T_1b734_row17_col4" class="data row17 col4" >-8.68%</td>
      <td id="T_1b734_row17_col5" class="data row17 col5" >0.85%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row18" class="row_heading level0 row18" >Cr</th>
      <td id="T_1b734_row18_col0" class="data row18 col0" >BCC</td>
      <td id="T_1b734_row18_col1" class="data row18 col1" >2.880000</td>
      <td id="T_1b734_row18_col2" class="data row18 col2" >2.968899</td>
      <td id="T_1b734_row18_col3" class="data row18 col3" >2.859300</td>
      <td id="T_1b734_row18_col4" class="data row18 col4" >-0.72%</td>
      <td id="T_1b734_row18_col5" class="data row18 col5" >-3.69%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row19" class="row_heading level0 row19" >CrN</th>
      <td id="T_1b734_row19_col0" class="data row19 col0" >Halite</td>
      <td id="T_1b734_row19_col1" class="data row19 col1" >4.149000</td>
      <td id="T_1b734_row19_col2" class="data row19 col2" >4.190855</td>
      <td id="T_1b734_row19_col3" class="data row19 col3" >4.142035</td>
      <td id="T_1b734_row19_col4" class="data row19 col4" >-0.17%</td>
      <td id="T_1b734_row19_col5" class="data row19 col5" >-1.16%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row20" class="row_heading level0 row20" >Cs</th>
      <td id="T_1b734_row20_col0" class="data row20 col0" >BCC</td>
      <td id="T_1b734_row20_col1" class="data row20 col1" >6.050000</td>
      <td id="T_1b734_row20_col2" class="data row20 col2" >6.256930</td>
      <td id="T_1b734_row20_col3" class="data row20 col3" >5.318184</td>
      <td id="T_1b734_row20_col4" class="data row20 col4" >-12.10%</td>
      <td id="T_1b734_row20_col5" class="data row20 col5" >-15.00%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row21" class="row_heading level0 row21" >CsCl</th>
      <td id="T_1b734_row21_col0" class="data row21 col0" >Caesium chloride</td>
      <td id="T_1b734_row21_col1" class="data row21 col1" >4.123000</td>
      <td id="T_1b734_row21_col2" class="data row21 col2" >4.143698</td>
      <td id="T_1b734_row21_col3" class="data row21 col3" >4.211410</td>
      <td id="T_1b734_row21_col4" class="data row21 col4" >2.14%</td>
      <td id="T_1b734_row21_col5" class="data row21 col5" >1.63%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row22" class="row_heading level0 row22" >CsF</th>
      <td id="T_1b734_row22_col0" class="data row22 col0" >Halite</td>
      <td id="T_1b734_row22_col1" class="data row22 col1" >6.020000</td>
      <td id="T_1b734_row22_col2" class="data row22 col2" >6.012279</td>
      <td id="T_1b734_row22_col3" class="data row22 col3" >6.135397</td>
      <td id="T_1b734_row22_col4" class="data row22 col4" >1.92%</td>
      <td id="T_1b734_row22_col5" class="data row22 col5" >2.05%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row23" class="row_heading level0 row23" >CsI</th>
      <td id="T_1b734_row23_col0" class="data row23 col0" >Caesium chloride</td>
      <td id="T_1b734_row23_col1" class="data row23 col1" >4.567000</td>
      <td id="T_1b734_row23_col2" class="data row23 col2" >4.665212</td>
      <td id="T_1b734_row23_col3" class="data row23 col3" >4.679304</td>
      <td id="T_1b734_row23_col4" class="data row23 col4" >2.46%</td>
      <td id="T_1b734_row23_col5" class="data row23 col5" >0.30%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row24" class="row_heading level0 row24" >Cu</th>
      <td id="T_1b734_row24_col0" class="data row24 col0" >FCC</td>
      <td id="T_1b734_row24_col1" class="data row24 col1" >3.597000</td>
      <td id="T_1b734_row24_col2" class="data row24 col2" >3.577431</td>
      <td id="T_1b734_row24_col3" class="data row24 col3" >3.616886</td>
      <td id="T_1b734_row24_col4" class="data row24 col4" >0.55%</td>
      <td id="T_1b734_row24_col5" class="data row24 col5" >1.10%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row25" class="row_heading level0 row25" >Eu</th>
      <td id="T_1b734_row25_col0" class="data row25 col0" >BCC</td>
      <td id="T_1b734_row25_col1" class="data row25 col1" >4.610000</td>
      <td id="T_1b734_row25_col2" class="data row25 col2" >4.487602</td>
      <td id="T_1b734_row25_col3" class="data row25 col3" >4.496991</td>
      <td id="T_1b734_row25_col4" class="data row25 col4" >-2.45%</td>
      <td id="T_1b734_row25_col5" class="data row25 col5" >0.21%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row26" class="row_heading level0 row26" >EuTiO3</th>
      <td id="T_1b734_row26_col0" class="data row26 col0" >Cubic perovskite</td>
      <td id="T_1b734_row26_col1" class="data row26 col1" >7.810000</td>
      <td id="T_1b734_row26_col2" class="data row26 col2" >0.000000</td>
      <td id="T_1b734_row26_col3" class="data row26 col3" >3.932595</td>
      <td id="T_1b734_row26_col4" class="data row26 col4" >-49.65%</td>
      <td id="T_1b734_row26_col5" class="data row26 col5" >inf%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row27" class="row_heading level0 row27" >Fe</th>
      <td id="T_1b734_row27_col0" class="data row27 col0" >BCC</td>
      <td id="T_1b734_row27_col1" class="data row27 col1" >2.856000</td>
      <td id="T_1b734_row27_col2" class="data row27 col2" >2.863035</td>
      <td id="T_1b734_row27_col3" class="data row27 col3" >2.838155</td>
      <td id="T_1b734_row27_col4" class="data row27 col4" >-0.62%</td>
      <td id="T_1b734_row27_col5" class="data row27 col5" >-0.87%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row28" class="row_heading level0 row28" >GaAs</th>
      <td id="T_1b734_row28_col0" class="data row28 col0" >Zinc blende (FCC)</td>
      <td id="T_1b734_row28_col1" class="data row28 col1" >5.653000</td>
      <td id="T_1b734_row28_col2" class="data row28 col2" >5.750182</td>
      <td id="T_1b734_row28_col3" class="data row28 col3" >5.744502</td>
      <td id="T_1b734_row28_col4" class="data row28 col4" >1.62%</td>
      <td id="T_1b734_row28_col5" class="data row28 col5" >-0.10%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row29" class="row_heading level0 row29" >GaP</th>
      <td id="T_1b734_row29_col0" class="data row29 col0" >Zinc blende (FCC)</td>
      <td id="T_1b734_row29_col1" class="data row29 col1" >5.450500</td>
      <td id="T_1b734_row29_col2" class="data row29 col2" >5.451625</td>
      <td id="T_1b734_row29_col3" class="data row29 col3" >5.488140</td>
      <td id="T_1b734_row29_col4" class="data row29 col4" >0.69%</td>
      <td id="T_1b734_row29_col5" class="data row29 col5" >0.67%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row30" class="row_heading level0 row30" >GaSb</th>
      <td id="T_1b734_row30_col0" class="data row30 col0" >Zinc blende (FCC)</td>
      <td id="T_1b734_row30_col1" class="data row30 col1" >6.095900</td>
      <td id="T_1b734_row30_col2" class="data row30 col2" >6.137209</td>
      <td id="T_1b734_row30_col3" class="data row30 col3" >6.228623</td>
      <td id="T_1b734_row30_col4" class="data row30 col4" >2.18%</td>
      <td id="T_1b734_row30_col5" class="data row30 col5" >1.49%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row31" class="row_heading level0 row31" >Ge</th>
      <td id="T_1b734_row31_col0" class="data row31 col0" >Diamond (FCC)</td>
      <td id="T_1b734_row31_col1" class="data row31 col1" >5.658000</td>
      <td id="T_1b734_row31_col2" class="data row31 col2" >5.674854</td>
      <td id="T_1b734_row31_col3" class="data row31 col3" >5.771785</td>
      <td id="T_1b734_row31_col4" class="data row31 col4" >2.01%</td>
      <td id="T_1b734_row31_col5" class="data row31 col5" >1.71%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row32" class="row_heading level0 row32" >HfC0.99</th>
      <td id="T_1b734_row32_col0" class="data row32 col0" >Halite</td>
      <td id="T_1b734_row32_col1" class="data row32 col1" >4.640000</td>
      <td id="T_1b734_row32_col2" class="data row32 col2" >4.634278</td>
      <td id="T_1b734_row32_col3" class="data row32 col3" >4.644392</td>
      <td id="T_1b734_row32_col4" class="data row32 col4" >0.09%</td>
      <td id="T_1b734_row32_col5" class="data row32 col5" >0.22%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row33" class="row_heading level0 row33" >HfN</th>
      <td id="T_1b734_row33_col0" class="data row33 col0" >Halite</td>
      <td id="T_1b734_row33_col1" class="data row33 col1" >4.392000</td>
      <td id="T_1b734_row33_col2" class="data row33 col2" >4.511720</td>
      <td id="T_1b734_row33_col3" class="data row33 col3" >4.535371</td>
      <td id="T_1b734_row33_col4" class="data row33 col4" >3.26%</td>
      <td id="T_1b734_row33_col5" class="data row33 col5" >0.52%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row34" class="row_heading level0 row34" >InAs</th>
      <td id="T_1b734_row34_col0" class="data row34 col0" >Zinc blende (FCC)</td>
      <td id="T_1b734_row34_col1" class="data row34 col1" >6.058300</td>
      <td id="T_1b734_row34_col2" class="data row34 col2" >6.107123</td>
      <td id="T_1b734_row34_col3" class="data row34 col3" >6.198173</td>
      <td id="T_1b734_row34_col4" class="data row34 col4" >2.31%</td>
      <td id="T_1b734_row34_col5" class="data row34 col5" >1.49%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row35" class="row_heading level0 row35" >InP</th>
      <td id="T_1b734_row35_col0" class="data row35 col0" >Zinc blende (FCC)</td>
      <td id="T_1b734_row35_col1" class="data row35 col1" >5.869000</td>
      <td id="T_1b734_row35_col2" class="data row35 col2" >5.903953</td>
      <td id="T_1b734_row35_col3" class="data row35 col3" >5.961731</td>
      <td id="T_1b734_row35_col4" class="data row35 col4" >1.58%</td>
      <td id="T_1b734_row35_col5" class="data row35 col5" >0.98%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row36" class="row_heading level0 row36" >InSb</th>
      <td id="T_1b734_row36_col0" class="data row36 col0" >Zinc blende (FCC)</td>
      <td id="T_1b734_row36_col1" class="data row36 col1" >6.479000</td>
      <td id="T_1b734_row36_col2" class="data row36 col2" >6.633221</td>
      <td id="T_1b734_row36_col3" class="data row36 col3" >6.653125</td>
      <td id="T_1b734_row36_col4" class="data row36 col4" >2.69%</td>
      <td id="T_1b734_row36_col5" class="data row36 col5" >0.30%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row37" class="row_heading level0 row37" >Ir</th>
      <td id="T_1b734_row37_col0" class="data row37 col0" >FCC</td>
      <td id="T_1b734_row37_col1" class="data row37 col1" >3.840000</td>
      <td id="T_1b734_row37_col2" class="data row37 col2" >3.853929</td>
      <td id="T_1b734_row37_col3" class="data row37 col3" >3.872626</td>
      <td id="T_1b734_row37_col4" class="data row37 col4" >0.85%</td>
      <td id="T_1b734_row37_col5" class="data row37 col5" >0.49%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row38" class="row_heading level0 row38" >K</th>
      <td id="T_1b734_row38_col0" class="data row38 col0" >BCC</td>
      <td id="T_1b734_row38_col1" class="data row38 col1" >5.230000</td>
      <td id="T_1b734_row38_col2" class="data row38 col2" >5.395117</td>
      <td id="T_1b734_row38_col3" class="data row38 col3" >5.377263</td>
      <td id="T_1b734_row38_col4" class="data row38 col4" >2.82%</td>
      <td id="T_1b734_row38_col5" class="data row38 col5" >-0.33%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row39" class="row_heading level0 row39" >KBr</th>
      <td id="T_1b734_row39_col0" class="data row39 col0" >Halite</td>
      <td id="T_1b734_row39_col1" class="data row39 col1" >6.600000</td>
      <td id="T_1b734_row39_col2" class="data row39 col2" >6.589082</td>
      <td id="T_1b734_row39_col3" class="data row39 col3" >6.698631</td>
      <td id="T_1b734_row39_col4" class="data row39 col4" >1.49%</td>
      <td id="T_1b734_row39_col5" class="data row39 col5" >1.66%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row40" class="row_heading level0 row40" >KCl</th>
      <td id="T_1b734_row40_col0" class="data row40 col0" >Halite</td>
      <td id="T_1b734_row40_col1" class="data row40 col1" >6.290000</td>
      <td id="T_1b734_row40_col2" class="data row40 col2" >6.283743</td>
      <td id="T_1b734_row40_col3" class="data row40 col3" >6.389114</td>
      <td id="T_1b734_row40_col4" class="data row40 col4" >1.58%</td>
      <td id="T_1b734_row40_col5" class="data row40 col5" >1.68%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row41" class="row_heading level0 row41" >KF</th>
      <td id="T_1b734_row41_col0" class="data row41 col0" >Halite</td>
      <td id="T_1b734_row41_col1" class="data row41 col1" >5.340000</td>
      <td id="T_1b734_row41_col2" class="data row41 col2" >5.308827</td>
      <td id="T_1b734_row41_col3" class="data row41 col3" >5.425351</td>
      <td id="T_1b734_row41_col4" class="data row41 col4" >1.60%</td>
      <td id="T_1b734_row41_col5" class="data row41 col5" >2.19%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row42" class="row_heading level0 row42" >KI</th>
      <td id="T_1b734_row42_col0" class="data row42 col0" >Halite</td>
      <td id="T_1b734_row42_col1" class="data row42 col1" >7.070000</td>
      <td id="T_1b734_row42_col2" class="data row42 col2" >7.084871</td>
      <td id="T_1b734_row42_col3" class="data row42 col3" >7.164688</td>
      <td id="T_1b734_row42_col4" class="data row42 col4" >1.34%</td>
      <td id="T_1b734_row42_col5" class="data row42 col5" >1.13%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row43" class="row_heading level0 row43" >KTaO3</th>
      <td id="T_1b734_row43_col0" class="data row43 col0" >Cubic perovskite</td>
      <td id="T_1b734_row43_col1" class="data row43 col1" >3.988500</td>
      <td id="T_1b734_row43_col2" class="data row43 col2" >0.000000</td>
      <td id="T_1b734_row43_col3" class="data row43 col3" >4.035274</td>
      <td id="T_1b734_row43_col4" class="data row43 col4" >1.17%</td>
      <td id="T_1b734_row43_col5" class="data row43 col5" >inf%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row44" class="row_heading level0 row44" >Li</th>
      <td id="T_1b734_row44_col0" class="data row44 col0" >BCC</td>
      <td id="T_1b734_row44_col1" class="data row44 col1" >3.490000</td>
      <td id="T_1b734_row44_col2" class="data row44 col2" >3.439312</td>
      <td id="T_1b734_row44_col3" class="data row44 col3" >3.444487</td>
      <td id="T_1b734_row44_col4" class="data row44 col4" >-1.30%</td>
      <td id="T_1b734_row44_col5" class="data row44 col5" >0.15%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row45" class="row_heading level0 row45" >LiBr</th>
      <td id="T_1b734_row45_col0" class="data row45 col0" >Halite</td>
      <td id="T_1b734_row45_col1" class="data row45 col1" >5.500000</td>
      <td id="T_1b734_row45_col2" class="data row45 col2" >5.445381</td>
      <td id="T_1b734_row45_col3" class="data row45 col3" >5.542445</td>
      <td id="T_1b734_row45_col4" class="data row45 col4" >0.77%</td>
      <td id="T_1b734_row45_col5" class="data row45 col5" >1.78%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row46" class="row_heading level0 row46" >LiCl</th>
      <td id="T_1b734_row46_col0" class="data row46 col0" >Halite</td>
      <td id="T_1b734_row46_col1" class="data row46 col1" >5.140000</td>
      <td id="T_1b734_row46_col2" class="data row46 col2" >5.084245</td>
      <td id="T_1b734_row46_col3" class="data row46 col3" >5.137603</td>
      <td id="T_1b734_row46_col4" class="data row46 col4" >-0.05%</td>
      <td id="T_1b734_row46_col5" class="data row46 col5" >1.05%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row47" class="row_heading level0 row47" >LiF</th>
      <td id="T_1b734_row47_col0" class="data row47 col0" >Halite</td>
      <td id="T_1b734_row47_col1" class="data row47 col1" >4.030000</td>
      <td id="T_1b734_row47_col2" class="data row47 col2" >4.083427</td>
      <td id="T_1b734_row47_col3" class="data row47 col3" >4.089020</td>
      <td id="T_1b734_row47_col4" class="data row47 col4" >1.46%</td>
      <td id="T_1b734_row47_col5" class="data row47 col5" >0.14%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row48" class="row_heading level0 row48" >LiI</th>
      <td id="T_1b734_row48_col0" class="data row48 col0" >Halite</td>
      <td id="T_1b734_row48_col1" class="data row48 col1" >6.010000</td>
      <td id="T_1b734_row48_col2" class="data row48 col2" >5.968355</td>
      <td id="T_1b734_row48_col3" class="data row48 col3" >6.053774</td>
      <td id="T_1b734_row48_col4" class="data row48 col4" >0.73%</td>
      <td id="T_1b734_row48_col5" class="data row48 col5" >1.43%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row49" class="row_heading level0 row49" >MgO</th>
      <td id="T_1b734_row49_col0" class="data row49 col0" >Halite (FCC)</td>
      <td id="T_1b734_row49_col1" class="data row49 col1" >4.212000</td>
      <td id="T_1b734_row49_col2" class="data row49 col2" >4.194003</td>
      <td id="T_1b734_row49_col3" class="data row49 col3" >4.264568</td>
      <td id="T_1b734_row49_col4" class="data row49 col4" >1.25%</td>
      <td id="T_1b734_row49_col5" class="data row49 col5" >1.68%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row50" class="row_heading level0 row50" >Mo</th>
      <td id="T_1b734_row50_col0" class="data row50 col0" >BCC</td>
      <td id="T_1b734_row50_col1" class="data row50 col1" >3.142000</td>
      <td id="T_1b734_row50_col2" class="data row50 col2" >3.167618</td>
      <td id="T_1b734_row50_col3" class="data row50 col3" >3.166742</td>
      <td id="T_1b734_row50_col4" class="data row50 col4" >0.79%</td>
      <td id="T_1b734_row50_col5" class="data row50 col5" >-0.03%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row51" class="row_heading level0 row51" >Na</th>
      <td id="T_1b734_row51_col0" class="data row51 col0" >BCC</td>
      <td id="T_1b734_row51_col1" class="data row51 col1" >4.230000</td>
      <td id="T_1b734_row51_col2" class="data row51 col2" >4.208054</td>
      <td id="T_1b734_row51_col3" class="data row51 col3" >4.226154</td>
      <td id="T_1b734_row51_col4" class="data row51 col4" >-0.09%</td>
      <td id="T_1b734_row51_col5" class="data row51 col5" >0.43%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row52" class="row_heading level0 row52" >NaBr</th>
      <td id="T_1b734_row52_col0" class="data row52 col0" >Halite</td>
      <td id="T_1b734_row52_col1" class="data row52 col1" >5.970000</td>
      <td id="T_1b734_row52_col2" class="data row52 col2" >5.923085</td>
      <td id="T_1b734_row52_col3" class="data row52 col3" >6.060649</td>
      <td id="T_1b734_row52_col4" class="data row52 col4" >1.52%</td>
      <td id="T_1b734_row52_col5" class="data row52 col5" >2.32%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row53" class="row_heading level0 row53" >NaCl</th>
      <td id="T_1b734_row53_col0" class="data row53 col0" >Halite</td>
      <td id="T_1b734_row53_col1" class="data row53 col1" >5.640000</td>
      <td id="T_1b734_row53_col2" class="data row53 col2" >5.588126</td>
      <td id="T_1b734_row53_col3" class="data row53 col3" >5.697209</td>
      <td id="T_1b734_row53_col4" class="data row53 col4" >1.01%</td>
      <td id="T_1b734_row53_col5" class="data row53 col5" >1.95%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row54" class="row_heading level0 row54" >NaF</th>
      <td id="T_1b734_row54_col0" class="data row54 col0" >Halite</td>
      <td id="T_1b734_row54_col1" class="data row54 col1" >4.630000</td>
      <td id="T_1b734_row54_col2" class="data row54 col2" >4.571789</td>
      <td id="T_1b734_row54_col3" class="data row54 col3" >4.716607</td>
      <td id="T_1b734_row54_col4" class="data row54 col4" >1.87%</td>
      <td id="T_1b734_row54_col5" class="data row54 col5" >3.17%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row55" class="row_heading level0 row55" >NaI</th>
      <td id="T_1b734_row55_col0" class="data row55 col0" >Halite</td>
      <td id="T_1b734_row55_col1" class="data row55 col1" >6.470000</td>
      <td id="T_1b734_row55_col2" class="data row55 col2" >6.437313</td>
      <td id="T_1b734_row55_col3" class="data row55 col3" >6.506101</td>
      <td id="T_1b734_row55_col4" class="data row55 col4" >0.56%</td>
      <td id="T_1b734_row55_col5" class="data row55 col5" >1.07%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row56" class="row_heading level0 row56" >Nb</th>
      <td id="T_1b734_row56_col0" class="data row56 col0" >BCC</td>
      <td id="T_1b734_row56_col1" class="data row56 col1" >3.300800</td>
      <td id="T_1b734_row56_col2" class="data row56 col2" >3.317632</td>
      <td id="T_1b734_row56_col3" class="data row56 col3" >3.317783</td>
      <td id="T_1b734_row56_col4" class="data row56 col4" >0.51%</td>
      <td id="T_1b734_row56_col5" class="data row56 col5" >0.00%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row57" class="row_heading level0 row57" >NbN</th>
      <td id="T_1b734_row57_col0" class="data row57 col0" >Halite</td>
      <td id="T_1b734_row57_col1" class="data row57 col1" >4.392000</td>
      <td id="T_1b734_row57_col2" class="data row57 col2" >4.452468</td>
      <td id="T_1b734_row57_col3" class="data row57 col3" >4.454411</td>
      <td id="T_1b734_row57_col4" class="data row57 col4" >1.42%</td>
      <td id="T_1b734_row57_col5" class="data row57 col5" >0.04%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row58" class="row_heading level0 row58" >Ne</th>
      <td id="T_1b734_row58_col0" class="data row58 col0" >FCC</td>
      <td id="T_1b734_row58_col1" class="data row58 col1" >4.430000</td>
      <td id="T_1b734_row58_col2" class="data row58 col2" >4.195020</td>
      <td id="T_1b734_row58_col3" class="data row58 col3" >4.338519</td>
      <td id="T_1b734_row58_col4" class="data row58 col4" >-2.07%</td>
      <td id="T_1b734_row58_col5" class="data row58 col5" >3.42%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row59" class="row_heading level0 row59" >Ni</th>
      <td id="T_1b734_row59_col0" class="data row59 col0" >FCC</td>
      <td id="T_1b734_row59_col1" class="data row59 col1" >3.499000</td>
      <td id="T_1b734_row59_col2" class="data row59 col2" >3.475146</td>
      <td id="T_1b734_row59_col3" class="data row59 col3" >3.503366</td>
      <td id="T_1b734_row59_col4" class="data row59 col4" >0.12%</td>
      <td id="T_1b734_row59_col5" class="data row59 col5" >0.81%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row60" class="row_heading level0 row60" >Pb</th>
      <td id="T_1b734_row60_col0" class="data row60 col0" >FCC</td>
      <td id="T_1b734_row60_col1" class="data row60 col1" >4.920000</td>
      <td id="T_1b734_row60_col2" class="data row60 col2" >4.989509</td>
      <td id="T_1b734_row60_col3" class="data row60 col3" >5.025697</td>
      <td id="T_1b734_row60_col4" class="data row60 col4" >2.15%</td>
      <td id="T_1b734_row60_col5" class="data row60 col5" >0.73%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row61" class="row_heading level0 row61" >PbS</th>
      <td id="T_1b734_row61_col0" class="data row61 col0" >Halite (FCC)</td>
      <td id="T_1b734_row61_col1" class="data row61 col1" >5.936200</td>
      <td id="T_1b734_row61_col2" class="data row61 col2" >0.000000</td>
      <td id="T_1b734_row61_col3" class="data row61 col3" >6.010332</td>
      <td id="T_1b734_row61_col4" class="data row61 col4" >1.25%</td>
      <td id="T_1b734_row61_col5" class="data row61 col5" >inf%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row62" class="row_heading level0 row62" >PbTe</th>
      <td id="T_1b734_row62_col0" class="data row62 col0" >Halite (FCC)</td>
      <td id="T_1b734_row62_col1" class="data row62 col1" >6.462000</td>
      <td id="T_1b734_row62_col2" class="data row62 col2" >6.541794</td>
      <td id="T_1b734_row62_col3" class="data row62 col3" >6.560341</td>
      <td id="T_1b734_row62_col4" class="data row62 col4" >1.52%</td>
      <td id="T_1b734_row62_col5" class="data row62 col5" >0.28%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row63" class="row_heading level0 row63" >Pd</th>
      <td id="T_1b734_row63_col0" class="data row63 col0" >FCC</td>
      <td id="T_1b734_row63_col1" class="data row63 col1" >3.859000</td>
      <td id="T_1b734_row63_col2" class="data row63 col2" >3.917302</td>
      <td id="T_1b734_row63_col3" class="data row63 col3" >3.938261</td>
      <td id="T_1b734_row63_col4" class="data row63 col4" >2.05%</td>
      <td id="T_1b734_row63_col5" class="data row63 col5" >0.54%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row64" class="row_heading level0 row64" >Pt</th>
      <td id="T_1b734_row64_col0" class="data row64 col0" >FCC</td>
      <td id="T_1b734_row64_col1" class="data row64 col1" >3.912000</td>
      <td id="T_1b734_row64_col2" class="data row64 col2" >3.943150</td>
      <td id="T_1b734_row64_col3" class="data row64 col3" >3.976443</td>
      <td id="T_1b734_row64_col4" class="data row64 col4" >1.65%</td>
      <td id="T_1b734_row64_col5" class="data row64 col5" >0.84%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row65" class="row_heading level0 row65" >RbBr</th>
      <td id="T_1b734_row65_col0" class="data row65 col0" >Halite</td>
      <td id="T_1b734_row65_col1" class="data row65 col1" >6.890000</td>
      <td id="T_1b734_row65_col2" class="data row65 col2" >6.911901</td>
      <td id="T_1b734_row65_col3" class="data row65 col3" >7.001376</td>
      <td id="T_1b734_row65_col4" class="data row65 col4" >1.62%</td>
      <td id="T_1b734_row65_col5" class="data row65 col5" >1.29%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row66" class="row_heading level0 row66" >RbCl</th>
      <td id="T_1b734_row66_col0" class="data row66 col0" >Halite</td>
      <td id="T_1b734_row66_col1" class="data row66 col1" >6.590000</td>
      <td id="T_1b734_row66_col2" class="data row66 col2" >6.617415</td>
      <td id="T_1b734_row66_col3" class="data row66 col3" >6.721173</td>
      <td id="T_1b734_row66_col4" class="data row66 col4" >1.99%</td>
      <td id="T_1b734_row66_col5" class="data row66 col5" >1.57%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row67" class="row_heading level0 row67" >RbF</th>
      <td id="T_1b734_row67_col0" class="data row67 col0" >Halite</td>
      <td id="T_1b734_row67_col1" class="data row67 col1" >5.650000</td>
      <td id="T_1b734_row67_col2" class="data row67 col2" >5.632284</td>
      <td id="T_1b734_row67_col3" class="data row67 col3" >5.722590</td>
      <td id="T_1b734_row67_col4" class="data row67 col4" >1.28%</td>
      <td id="T_1b734_row67_col5" class="data row67 col5" >1.60%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row68" class="row_heading level0 row68" >RbI</th>
      <td id="T_1b734_row68_col0" class="data row68 col0" >Halite</td>
      <td id="T_1b734_row68_col1" class="data row68 col1" >7.350000</td>
      <td id="T_1b734_row68_col2" class="data row68 col2" >7.382335</td>
      <td id="T_1b734_row68_col3" class="data row68 col3" >7.525812</td>
      <td id="T_1b734_row68_col4" class="data row68 col4" >2.39%</td>
      <td id="T_1b734_row68_col5" class="data row68 col5" >1.94%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row69" class="row_heading level0 row69" >Rh</th>
      <td id="T_1b734_row69_col0" class="data row69 col0" >FCC</td>
      <td id="T_1b734_row69_col1" class="data row69 col1" >3.800000</td>
      <td id="T_1b734_row69_col2" class="data row69 col2" >3.805972</td>
      <td id="T_1b734_row69_col3" class="data row69 col3" >3.846733</td>
      <td id="T_1b734_row69_col4" class="data row69 col4" >1.23%</td>
      <td id="T_1b734_row69_col5" class="data row69 col5" >1.07%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row70" class="row_heading level0 row70" >ScN</th>
      <td id="T_1b734_row70_col0" class="data row70 col0" >Halite</td>
      <td id="T_1b734_row70_col1" class="data row70 col1" >4.520000</td>
      <td id="T_1b734_row70_col2" class="data row70 col2" >4.510944</td>
      <td id="T_1b734_row70_col3" class="data row70 col3" >4.516923</td>
      <td id="T_1b734_row70_col4" class="data row70 col4" >-0.07%</td>
      <td id="T_1b734_row70_col5" class="data row70 col5" >0.13%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row71" class="row_heading level0 row71" >Si</th>
      <td id="T_1b734_row71_col0" class="data row71 col0" >Diamond (FCC)</td>
      <td id="T_1b734_row71_col1" class="data row71 col1" >5.431021</td>
      <td id="T_1b734_row71_col2" class="data row71 col2" >5.443702</td>
      <td id="T_1b734_row71_col3" class="data row71 col3" >5.458414</td>
      <td id="T_1b734_row71_col4" class="data row71 col4" >0.50%</td>
      <td id="T_1b734_row71_col5" class="data row71 col5" >0.27%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row72" class="row_heading level0 row72" >Sr</th>
      <td id="T_1b734_row72_col0" class="data row72 col0" >FCC</td>
      <td id="T_1b734_row72_col1" class="data row72 col1" >6.080000</td>
      <td id="T_1b734_row72_col2" class="data row72 col2" >6.067206</td>
      <td id="T_1b734_row72_col3" class="data row72 col3" >6.054884</td>
      <td id="T_1b734_row72_col4" class="data row72 col4" >-0.41%</td>
      <td id="T_1b734_row72_col5" class="data row72 col5" >-0.20%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row73" class="row_heading level0 row73" >SrTiO3</th>
      <td id="T_1b734_row73_col0" class="data row73 col0" >Cubic perovskite</td>
      <td id="T_1b734_row73_col1" class="data row73 col1" >3.988050</td>
      <td id="T_1b734_row73_col2" class="data row73 col2" >3.912701</td>
      <td id="T_1b734_row73_col3" class="data row73 col3" >3.944839</td>
      <td id="T_1b734_row73_col4" class="data row73 col4" >-1.08%</td>
      <td id="T_1b734_row73_col5" class="data row73 col5" >0.82%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row74" class="row_heading level0 row74" >SrVO3</th>
      <td id="T_1b734_row74_col0" class="data row74 col0" >Cubic perovskite</td>
      <td id="T_1b734_row74_col1" class="data row74 col1" >3.838000</td>
      <td id="T_1b734_row74_col2" class="data row74 col2" >3.900891</td>
      <td id="T_1b734_row74_col3" class="data row74 col3" >3.911846</td>
      <td id="T_1b734_row74_col4" class="data row74 col4" >1.92%</td>
      <td id="T_1b734_row74_col5" class="data row74 col5" >0.28%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row75" class="row_heading level0 row75" >Ta</th>
      <td id="T_1b734_row75_col0" class="data row75 col0" >BCC</td>
      <td id="T_1b734_row75_col1" class="data row75 col1" >3.305800</td>
      <td id="T_1b734_row75_col2" class="data row75 col2" >3.309856</td>
      <td id="T_1b734_row75_col3" class="data row75 col3" >3.326660</td>
      <td id="T_1b734_row75_col4" class="data row75 col4" >0.63%</td>
      <td id="T_1b734_row75_col5" class="data row75 col5" >0.51%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row76" class="row_heading level0 row76" >TaC0.99</th>
      <td id="T_1b734_row76_col0" class="data row76 col0" >Halite</td>
      <td id="T_1b734_row76_col1" class="data row76 col1" >4.456000</td>
      <td id="T_1b734_row76_col2" class="data row76 col2" >4.467795</td>
      <td id="T_1b734_row76_col3" class="data row76 col3" >4.474708</td>
      <td id="T_1b734_row76_col4" class="data row76 col4" >0.42%</td>
      <td id="T_1b734_row76_col5" class="data row76 col5" >0.15%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row77" class="row_heading level0 row77" >Th</th>
      <td id="T_1b734_row77_col0" class="data row77 col0" >FCC</td>
      <td id="T_1b734_row77_col1" class="data row77 col1" >5.080000</td>
      <td id="T_1b734_row77_col2" class="data row77 col2" >5.047255</td>
      <td id="T_1b734_row77_col3" class="data row77 col3" >5.055762</td>
      <td id="T_1b734_row77_col4" class="data row77 col4" >-0.48%</td>
      <td id="T_1b734_row77_col5" class="data row77 col5" >0.17%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row78" class="row_heading level0 row78" >TiC</th>
      <td id="T_1b734_row78_col0" class="data row78 col0" >Halite</td>
      <td id="T_1b734_row78_col1" class="data row78 col1" >4.328000</td>
      <td id="T_1b734_row78_col2" class="data row78 col2" >4.331438</td>
      <td id="T_1b734_row78_col3" class="data row78 col3" >4.337783</td>
      <td id="T_1b734_row78_col4" class="data row78 col4" >0.23%</td>
      <td id="T_1b734_row78_col5" class="data row78 col5" >0.15%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row79" class="row_heading level0 row79" >TiN</th>
      <td id="T_1b734_row79_col0" class="data row79 col0" >Halite</td>
      <td id="T_1b734_row79_col1" class="data row79 col1" >4.249000</td>
      <td id="T_1b734_row79_col2" class="data row79 col2" >4.241247</td>
      <td id="T_1b734_row79_col3" class="data row79 col3" >4.246024</td>
      <td id="T_1b734_row79_col4" class="data row79 col4" >-0.07%</td>
      <td id="T_1b734_row79_col5" class="data row79 col5" >0.11%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row80" class="row_heading level0 row80" >V</th>
      <td id="T_1b734_row80_col0" class="data row80 col0" >BCC</td>
      <td id="T_1b734_row80_col1" class="data row80 col1" >3.039900</td>
      <td id="T_1b734_row80_col2" class="data row80 col2" >2.982399</td>
      <td id="T_1b734_row80_col3" class="data row80 col3" >2.985185</td>
      <td id="T_1b734_row80_col4" class="data row80 col4" >-1.80%</td>
      <td id="T_1b734_row80_col5" class="data row80 col5" >0.09%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row81" class="row_heading level0 row81" >VC0.97</th>
      <td id="T_1b734_row81_col0" class="data row81 col0" >Halite</td>
      <td id="T_1b734_row81_col1" class="data row81 col1" >4.166000</td>
      <td id="T_1b734_row81_col2" class="data row81 col2" >4.161946</td>
      <td id="T_1b734_row81_col3" class="data row81 col3" >4.145138</td>
      <td id="T_1b734_row81_col4" class="data row81 col4" >-0.50%</td>
      <td id="T_1b734_row81_col5" class="data row81 col5" >-0.40%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row82" class="row_heading level0 row82" >VN</th>
      <td id="T_1b734_row82_col0" class="data row82 col0" >Halite</td>
      <td id="T_1b734_row82_col1" class="data row82 col1" >4.136000</td>
      <td id="T_1b734_row82_col2" class="data row82 col2" >4.124930</td>
      <td id="T_1b734_row82_col3" class="data row82 col3" >4.131446</td>
      <td id="T_1b734_row82_col4" class="data row82 col4" >-0.11%</td>
      <td id="T_1b734_row82_col5" class="data row82 col5" >0.16%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row83" class="row_heading level0 row83" >W</th>
      <td id="T_1b734_row83_col0" class="data row83 col0" >BCC</td>
      <td id="T_1b734_row83_col1" class="data row83 col1" >3.155000</td>
      <td id="T_1b734_row83_col2" class="data row83 col2" >3.170316</td>
      <td id="T_1b734_row83_col3" class="data row83 col3" >3.178230</td>
      <td id="T_1b734_row83_col4" class="data row83 col4" >0.74%</td>
      <td id="T_1b734_row83_col5" class="data row83 col5" >0.25%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row84" class="row_heading level0 row84" >Yb</th>
      <td id="T_1b734_row84_col0" class="data row84 col0" >FCC</td>
      <td id="T_1b734_row84_col1" class="data row84 col1" >5.490000</td>
      <td id="T_1b734_row84_col2" class="data row84 col2" >5.387257</td>
      <td id="T_1b734_row84_col3" class="data row84 col3" >5.483943</td>
      <td id="T_1b734_row84_col4" class="data row84 col4" >-0.11%</td>
      <td id="T_1b734_row84_col5" class="data row84 col5" >1.79%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row85" class="row_heading level0 row85" >ZnO</th>
      <td id="T_1b734_row85_col0" class="data row85 col0" >Halite (FCC)</td>
      <td id="T_1b734_row85_col1" class="data row85 col1" >4.580000</td>
      <td id="T_1b734_row85_col2" class="data row85 col2" >4.338884</td>
      <td id="T_1b734_row85_col3" class="data row85 col3" >4.334630</td>
      <td id="T_1b734_row85_col4" class="data row85 col4" >-5.36%</td>
      <td id="T_1b734_row85_col5" class="data row85 col5" >-0.10%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row86" class="row_heading level0 row86" >ZnS</th>
      <td id="T_1b734_row86_col0" class="data row86 col0" >Zinc blende (FCC)</td>
      <td id="T_1b734_row86_col1" class="data row86 col1" >5.420000</td>
      <td id="T_1b734_row86_col2" class="data row86 col2" >5.387366</td>
      <td id="T_1b734_row86_col3" class="data row86 col3" >5.453956</td>
      <td id="T_1b734_row86_col4" class="data row86 col4" >0.63%</td>
      <td id="T_1b734_row86_col5" class="data row86 col5" >1.24%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row87" class="row_heading level0 row87" >ZrC0.97</th>
      <td id="T_1b734_row87_col0" class="data row87 col0" >Halite</td>
      <td id="T_1b734_row87_col1" class="data row87 col1" >4.698000</td>
      <td id="T_1b734_row87_col2" class="data row87 col2" >4.712866</td>
      <td id="T_1b734_row87_col3" class="data row87 col3" >4.717641</td>
      <td id="T_1b734_row87_col4" class="data row87 col4" >0.42%</td>
      <td id="T_1b734_row87_col5" class="data row87 col5" >0.10%</td>
    </tr>
    <tr>
      <th id="T_1b734_level0_row88" class="row_heading level0 row88" >ZrN</th>
      <td id="T_1b734_row88_col0" class="data row88 col0" >Halite</td>
      <td id="T_1b734_row88_col1" class="data row88 col1" >4.577000</td>
      <td id="T_1b734_row88_col2" class="data row88 col2" >4.588531</td>
      <td id="T_1b734_row88_col3" class="data row88 col3" >4.612292</td>
      <td id="T_1b734_row88_col4" class="data row88 col4" >0.77%</td>
      <td id="T_1b734_row88_col5" class="data row88 col5" >0.52%</td>
    </tr>
  </tbody>
</table>





```python
data["% error vs MP"].replace([np.inf, -np.inf], np.nan).dropna().hist(bins=20)
```




    <Axes: >




    
![png](Errors%20in%20Cubic%20Crystal%20Lattice%20Parameters_files/Errors%20in%20Cubic%20Crystal%20Lattice%20Parameters_9_1.png)
    



```python
# This generates a pretty markdown table output.

# df = data.sort_values("% error vs MP", key=abs).replace([np.inf, -np.inf], np.nan).dropna()
# df["% error vs MP"] = [f"{v*100:.3f}%" for v in df["% error vs MP"]]
# df["% error vs Expt"] = [f"{v*100:.3f}%" for v in df["% error vs Expt"]]
# print(df.to_markdown())
```
