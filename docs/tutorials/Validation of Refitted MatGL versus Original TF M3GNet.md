---
layout: default
title: Validation of Refitted MatGL versus Original TF M3GNet.md
nav_exclude: true
---

# Introduction

In this notebook, we validate the refitted MatGL implementation of the M3GNet universal potential against the original TF version using cubic crystals. This is purely for convenience and demonstration purposes. The test can be carried out on any crystal. If you are running this in Google Colab, please uncomment the following lines.



```python
# !pip install m3gnet
# !pip install matgl
# !pip install pymatgen
# !pip install --upgrade mp-api
```


```python
from __future__ import annotations

import os
import warnings

import pandas as pd
from m3gnet.models import M3GNet as M3GNet_tf
from m3gnet.models import M3GNetCalculator as M3GNetCalculator_tf
from m3gnet.models import Potential as Potential_tf
from pymatgen.core import Composition, Lattice, Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import matgl
from matgl.ext.ase import M3GNetCalculator as M3GNetCalculator_matgl

warnings.filterwarnings("ignore")
```


```python
data = pd.read_html("http://en.wikipedia.org/wiki/Lattice_constant")[0]
struct_types = [
    "Hexagonal",
    "Wurtzite",
    "Wurtzite (HCP)",
    "Orthorhombic",
    "Tetragonal perovskite",
    "Orthorhombic perovskite",
]
data = data[~data["Crystal structure"].isin(struct_types)]
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


Let's load the old and new M3GNet models.



```python
# MatGL M3Gnet
potential_matgl = matgl.load_model("M3GNet-MP-2021.2.8-PES")
calc_matgl = M3GNetCalculator_matgl(potential_matgl)

# TF M3Gnet
model_tf = M3GNet_tf.load("MP-2021.2.8-EFS")
potential_tf = Potential_tf(model_tf)
calc_tf = M3GNetCalculator_tf(potential_tf)

adapter = AseAtomsAdaptor()
```


```python
os.environ["MPRESTER_MUTE_PROGRESS_BARS"] = "true"  # Mute progress bars for cleaner output.
mpr = MPRester()

structures = {}
results = []

for formula, v in data.iterrows():
    formula = formula.split()[0]
    c = Composition(formula)
    els = sorted(c.elements)
    cs = v["Crystal structure"]

    # We initialize all the crystals with an arbitrary lattice constant of 5 angstroms.
    if "Zinc blende" in cs:
        s = Structure.from_spacegroup("F-43m", Lattice.cubic(5), [els[0], els[1]], [[0, 0, 0], [0.25, 0.25, 0.75]])
    elif "Halite" in cs:
        s = Structure.from_spacegroup("Fm-3m", Lattice.cubic(5), [els[0], els[1]], [[0, 0, 0], [0.5, 0, 0]])
    elif "Caesium chloride" in cs:
        s = Structure.from_spacegroup("Pm-3m", Lattice.cubic(5), [els[0], els[1]], [[0, 0, 0], [0.5, 0.5, 0.5]])
    elif "Cubic perovskite" in cs:
        s = Structure(
            Lattice.cubic(5),
            [els[0], els[1], els[2], els[2], els[2]],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.0, 0.5, 0.5], [0.5, 0, 0.5]],
        )
    elif "Diamond" in cs:
        s = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5), [els[0]], [[0.25, 0.75, 0.25]])
    elif "BCC" in cs:
        s = Structure(Lattice.cubic(5), [els[0]] * 2, [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    elif "FCC" in cs:
        s = Structure(Lattice.cubic(5), [els[0]] * 4, [[0.0, 0.0, 0.0], [0.5, 0.5, 0], [0.0, 0.5, 0.5], [0.5, 0, 0.5]])
    else:
        continue
    try:
        mids = mpr.get_material_ids(s.composition.reduced_formula)
        for i in mids:
            try:
                structure = mpr.get_structure_by_material_id(i)
                sga = SpacegroupAnalyzer(structure)
                sga2 = SpacegroupAnalyzer(s)
                if sga.get_space_group_number() == sga2.get_space_group_number():
                    conv = sga.get_conventional_standard_structure()
                    structures[s.composition.reduced_formula] = conv
                    atoms_tf = adapter.get_atoms(conv)
                    atoms_matgl = adapter.get_atoms(conv)
                    atoms_tf.calc = calc_tf
                    atoms_matgl.calc = calc_matgl
                    results.append(
                        [
                            s.composition.reduced_formula,
                            atoms_tf.get_potential_energy() / len(s),
                            float(atoms_matgl.get_potential_energy() / len(s)),
                        ]
                    )
                    break
            except Exception:
                pass
    except Exception:
        pass
```

    2023-06-16 05:52:27.360487: W tensorflow/core/platform/profile_utils/cpu_utils.cc:128] Failed to get CPU frequency: 0 Hz
    Isolated atoms found in the structure



```python
df = pd.DataFrame(results, columns=["Formula", "MatGL Energy per Atom", "TF Energy per Atom"])
df["Difference"] = df["MatGL Energy per Atom"] - df["TF Energy per Atom"]
df = df.set_index("Formula")
df.sort_index().style.background_gradient()
```




<style type="text/css">
#T_1c633_row0_col0, #T_1c633_row5_col0, #T_1c633_row22_col0, #T_1c633_row63_col0 {
  background-color: #1e80b8;
  color: #f1f1f1;
}
#T_1c633_row0_col1, #T_1c633_row5_col1, #T_1c633_row22_col1, #T_1c633_row63_col1, #T_1c633_row84_col2 {
  background-color: #2081b9;
  color: #f1f1f1;
}
#T_1c633_row0_col2, #T_1c633_row15_col2 {
  background-color: #76aad0;
  color: #f1f1f1;
}
#T_1c633_row1_col0, #T_1c633_row1_col1, #T_1c633_row14_col0, #T_1c633_row40_col1 {
  background-color: #056aa6;
  color: #f1f1f1;
}
#T_1c633_row1_col2, #T_1c633_row12_col0, #T_1c633_row70_col1, #T_1c633_row82_col2 {
  background-color: #8eb3d5;
  color: #000000;
}
#T_1c633_row2_col0, #T_1c633_row28_col0, #T_1c633_row44_col1 {
  background-color: #1379b5;
  color: #f1f1f1;
}
#T_1c633_row2_col1, #T_1c633_row32_col0, #T_1c633_row73_col0 {
  background-color: #167bb6;
  color: #f1f1f1;
}
#T_1c633_row2_col2, #T_1c633_row31_col2, #T_1c633_row76_col2 {
  background-color: #509ac6;
  color: #f1f1f1;
}
#T_1c633_row3_col0, #T_1c633_row29_col1 {
  background-color: #2f8bbe;
  color: #f1f1f1;
}
#T_1c633_row3_col1, #T_1c633_row42_col2 {
  background-color: #328dbf;
  color: #f1f1f1;
}
#T_1c633_row3_col2, #T_1c633_row14_col1 {
  background-color: #056ba7;
  color: #f1f1f1;
}
#T_1c633_row4_col0, #T_1c633_row59_col0, #T_1c633_row59_col1 {
  background-color: #4295c3;
  color: #f1f1f1;
}
#T_1c633_row4_col1, #T_1c633_row60_col2 {
  background-color: #4496c3;
  color: #f1f1f1;
}
#T_1c633_row4_col2 {
  background-color: #3991c1;
  color: #f1f1f1;
}
#T_1c633_row5_col2, #T_1c633_row12_col2, #T_1c633_row14_col2, #T_1c633_row74_col0 {
  background-color: #91b5d6;
  color: #000000;
}
#T_1c633_row6_col0, #T_1c633_row23_col2, #T_1c633_row56_col0, #T_1c633_row56_col1 {
  background-color: #023858;
  color: #f1f1f1;
}
#T_1c633_row6_col1 {
  background-color: #02395a;
  color: #f1f1f1;
}
#T_1c633_row6_col2, #T_1c633_row37_col2 {
  background-color: #69a5cc;
  color: #f1f1f1;
}
#T_1c633_row7_col0 {
  background-color: #0570b0;
  color: #f1f1f1;
}
#T_1c633_row7_col1, #T_1c633_row19_col1, #T_1c633_row43_col0, #T_1c633_row62_col0 {
  background-color: #0771b1;
  color: #f1f1f1;
}
#T_1c633_row7_col2, #T_1c633_row9_col1, #T_1c633_row24_col2, #T_1c633_row70_col2 {
  background-color: #75a9cf;
  color: #f1f1f1;
}
#T_1c633_row8_col0, #T_1c633_row47_col2 {
  background-color: #b5c4df;
  color: #000000;
}
#T_1c633_row8_col1 {
  background-color: #b7c5df;
  color: #000000;
}
#T_1c633_row8_col2, #T_1c633_row10_col1, #T_1c633_row77_col0, #T_1c633_row77_col1 {
  background-color: #c1cae2;
  color: #000000;
}
#T_1c633_row9_col0 {
  background-color: #73a9cf;
  color: #f1f1f1;
}
#T_1c633_row9_col2 {
  background-color: #65a3cb;
  color: #f1f1f1;
}
#T_1c633_row10_col0 {
  background-color: #c0c9e2;
  color: #000000;
}
#T_1c633_row10_col2, #T_1c633_row41_col2, #T_1c633_row52_col2 {
  background-color: #a1bbda;
  color: #000000;
}
#T_1c633_row11_col0, #T_1c633_row42_col1 {
  background-color: #045e94;
  color: #f1f1f1;
}
#T_1c633_row11_col1 {
  background-color: #045f95;
  color: #f1f1f1;
}
#T_1c633_row11_col2 {
  background-color: #71a8ce;
  color: #f1f1f1;
}
#T_1c633_row12_col1, #T_1c633_row65_col0, #T_1c633_row65_col1 {
  background-color: #8fb4d6;
  color: #000000;
}
#T_1c633_row13_col0, #T_1c633_row37_col0, #T_1c633_row61_col1 {
  background-color: #056ead;
  color: #f1f1f1;
}
#T_1c633_row13_col1, #T_1c633_row19_col0 {
  background-color: #056faf;
  color: #f1f1f1;
}
#T_1c633_row13_col2 {
  background-color: #3d93c2;
  color: #f1f1f1;
}
#T_1c633_row15_col0, #T_1c633_row16_col2 {
  background-color: #05659f;
  color: #f1f1f1;
}
#T_1c633_row15_col1 {
  background-color: #0566a0;
  color: #f1f1f1;
}
#T_1c633_row16_col0 {
  background-color: #5c9fc9;
  color: #f1f1f1;
}
#T_1c633_row16_col1, #T_1c633_row44_col2, #T_1c633_row47_col1 {
  background-color: #62a2cb;
  color: #f1f1f1;
}
#T_1c633_row17_col0 {
  background-color: #cccfe5;
  color: #000000;
}
#T_1c633_row17_col1, #T_1c633_row75_col1 {
  background-color: #cacee5;
  color: #000000;
}
#T_1c633_row17_col2, #T_1c633_row31_col1 {
  background-color: #e6e2ef;
  color: #000000;
}
#T_1c633_row18_col0, #T_1c633_row18_col1, #T_1c633_row66_col1 {
  background-color: #c6cce3;
  color: #000000;
}
#T_1c633_row18_col2, #T_1c633_row24_col0, #T_1c633_row79_col1, #T_1c633_row84_col1 {
  background-color: #d1d2e6;
  color: #000000;
}
#T_1c633_row19_col2 {
  background-color: #5ea0ca;
  color: #f1f1f1;
}
#T_1c633_row20_col0 {
  background-color: #1b7eb7;
  color: #f1f1f1;
}
#T_1c633_row20_col1, #T_1c633_row27_col2 {
  background-color: #1c7fb8;
  color: #f1f1f1;
}
#T_1c633_row20_col2, #T_1c633_row53_col2, #T_1c633_row67_col2, #T_1c633_row83_col2 {
  background-color: #7dacd1;
  color: #f1f1f1;
}
#T_1c633_row21_col0, #T_1c633_row21_col1, #T_1c633_row40_col0, #T_1c633_row64_col1 {
  background-color: #0569a5;
  color: #f1f1f1;
}
#T_1c633_row21_col2 {
  background-color: #9fbad9;
  color: #000000;
}
#T_1c633_row22_col2, #T_1c633_row40_col2, #T_1c633_row54_col2 {
  background-color: #9ebad9;
  color: #000000;
}
#T_1c633_row23_col0 {
  background-color: #d9d8ea;
  color: #000000;
}
#T_1c633_row23_col1 {
  background-color: #dcdaeb;
  color: #000000;
}
#T_1c633_row24_col1, #T_1c633_row76_col0 {
  background-color: #d2d2e7;
  color: #000000;
}
#T_1c633_row25_col0, #T_1c633_row25_col1 {
  background-color: #b0c2de;
  color: #000000;
}
#T_1c633_row25_col2, #T_1c633_row35_col0, #T_1c633_row35_col1 {
  background-color: #bcc7e1;
  color: #000000;
}
#T_1c633_row26_col0, #T_1c633_row39_col0 {
  background-color: #2182b9;
  color: #f1f1f1;
}
#T_1c633_row26_col1, #T_1c633_row39_col1 {
  background-color: #2383ba;
  color: #f1f1f1;
}
#T_1c633_row26_col2, #T_1c633_row51_col2 {
  background-color: #89b1d4;
  color: #000000;
}
#T_1c633_row27_col0, #T_1c633_row29_col0 {
  background-color: #2d8abd;
  color: #f1f1f1;
}
#T_1c633_row27_col1, #T_1c633_row75_col2 {
  background-color: #308cbe;
  color: #f1f1f1;
}
#T_1c633_row28_col1, #T_1c633_row58_col1 {
  background-color: #157ab5;
  color: #f1f1f1;
}
#T_1c633_row28_col2 {
  background-color: #97b7d7;
  color: #000000;
}
#T_1c633_row29_col2, #T_1c633_row57_col2 {
  background-color: #9ab8d8;
  color: #000000;
}
#T_1c633_row30_col0 {
  background-color: #dfddec;
  color: #000000;
}
#T_1c633_row30_col1, #T_1c633_row65_col2 {
  background-color: #e0dded;
  color: #000000;
}
#T_1c633_row30_col2, #T_1c633_row61_col2 {
  background-color: #79abd0;
  color: #f1f1f1;
}
#T_1c633_row31_col0 {
  background-color: #e5e1ef;
  color: #000000;
}
#T_1c633_row32_col1 {
  background-color: #197db7;
  color: #f1f1f1;
}
#T_1c633_row32_col2, #T_1c633_row44_col0, #T_1c633_row58_col0 {
  background-color: #1278b4;
  color: #f1f1f1;
}
#T_1c633_row33_col0, #T_1c633_row52_col0, #T_1c633_row72_col2 {
  background-color: #2484ba;
  color: #f1f1f1;
}
#T_1c633_row33_col1, #T_1c633_row52_col1 {
  background-color: #2685bb;
  color: #f1f1f1;
}
#T_1c633_row33_col2 {
  background-color: #529bc7;
  color: #f1f1f1;
}
#T_1c633_row34_col0, #T_1c633_row83_col1 {
  background-color: #0d75b3;
  color: #f1f1f1;
}
#T_1c633_row34_col1 {
  background-color: #0f76b3;
  color: #f1f1f1;
}
#T_1c633_row34_col2 {
  background-color: #94b6d7;
  color: #000000;
}
#T_1c633_row35_col2, #T_1c633_row54_col0, #T_1c633_row85_col0 {
  background-color: #d6d6e9;
  color: #000000;
}
#T_1c633_row36_col0 {
  background-color: #034e7b;
  color: #f1f1f1;
}
#T_1c633_row36_col1 {
  background-color: #034f7d;
  color: #f1f1f1;
}
#T_1c633_row36_col2 {
  background-color: #6fa7ce;
  color: #f1f1f1;
}
#T_1c633_row37_col1 {
  background-color: #056fae;
  color: #f1f1f1;
}
#T_1c633_row38_col0, #T_1c633_row51_col1 {
  background-color: #0a73b2;
  color: #f1f1f1;
}
#T_1c633_row38_col1, #T_1c633_row83_col0 {
  background-color: #0c74b2;
  color: #f1f1f1;
}
#T_1c633_row38_col2, #T_1c633_row46_col2, #T_1c633_row66_col2 {
  background-color: #78abd0;
  color: #f1f1f1;
}
#T_1c633_row39_col2, #T_1c633_row63_col2, #T_1c633_row79_col2 {
  background-color: #88b1d4;
  color: #000000;
}
#T_1c633_row41_col0, #T_1c633_row41_col1 {
  background-color: #a9bfdc;
  color: #000000;
}
#T_1c633_row42_col0 {
  background-color: #045e93;
  color: #f1f1f1;
}
#T_1c633_row43_col1, #T_1c633_row51_col0, #T_1c633_row62_col1 {
  background-color: #0872b1;
  color: #f1f1f1;
}
#T_1c633_row43_col2 {
  background-color: #86b0d3;
  color: #000000;
}
#T_1c633_row45_col0 {
  background-color: #358fc0;
  color: #f1f1f1;
}
#T_1c633_row45_col1 {
  background-color: #3790c0;
  color: #f1f1f1;
}
#T_1c633_row45_col2, #T_1c633_row69_col2 {
  background-color: #7eadd1;
  color: #f1f1f1;
}
#T_1c633_row46_col0 {
  background-color: #056ba9;
  color: #f1f1f1;
}
#T_1c633_row46_col1 {
  background-color: #056caa;
  color: #f1f1f1;
}
#T_1c633_row47_col0 {
  background-color: #60a1ca;
  color: #f1f1f1;
}
#T_1c633_row48_col0 {
  background-color: #e4e1ef;
  color: #000000;
}
#T_1c633_row48_col1 {
  background-color: #e3e0ee;
  color: #000000;
}
#T_1c633_row48_col2, #T_1c633_row80_col0, #T_1c633_row80_col1 {
  background-color: #fff7fb;
  color: #000000;
}
#T_1c633_row49_col0 {
  background-color: #045382;
  color: #f1f1f1;
}
#T_1c633_row49_col1 {
  background-color: #045483;
  color: #f1f1f1;
}
#T_1c633_row49_col2 {
  background-color: #7bacd1;
  color: #f1f1f1;
}
#T_1c633_row50_col0 {
  background-color: #056dab;
  color: #f1f1f1;
}
#T_1c633_row50_col1, #T_1c633_row61_col0 {
  background-color: #056dac;
  color: #f1f1f1;
}
#T_1c633_row50_col2, #T_1c633_row67_col1 {
  background-color: #4a98c5;
  color: #f1f1f1;
}
#T_1c633_row53_col0 {
  background-color: #0568a3;
  color: #f1f1f1;
}
#T_1c633_row53_col1, #T_1c633_row64_col0 {
  background-color: #0569a4;
  color: #f1f1f1;
}
#T_1c633_row54_col1, #T_1c633_row55_col0 {
  background-color: #d7d6e9;
  color: #000000;
}
#T_1c633_row55_col1, #T_1c633_row85_col1 {
  background-color: #d8d7e9;
  color: #000000;
}
#T_1c633_row55_col2, #T_1c633_row60_col0 {
  background-color: #63a2cb;
  color: #f1f1f1;
}
#T_1c633_row56_col2 {
  background-color: #c9cee4;
  color: #000000;
}
#T_1c633_row57_col0 {
  background-color: #589ec8;
  color: #f1f1f1;
}
#T_1c633_row57_col1 {
  background-color: #5a9ec9;
  color: #f1f1f1;
}
#T_1c633_row58_col2 {
  background-color: #549cc7;
  color: #f1f1f1;
}
#T_1c633_row59_col2 {
  background-color: #d2d3e7;
  color: #000000;
}
#T_1c633_row60_col1 {
  background-color: #67a4cc;
  color: #f1f1f1;
}
#T_1c633_row62_col2 {
  background-color: #81aed2;
  color: #f1f1f1;
}
#T_1c633_row64_col2 {
  background-color: #abbfdc;
  color: #000000;
}
#T_1c633_row66_col0 {
  background-color: #c4cbe3;
  color: #000000;
}
#T_1c633_row67_col0 {
  background-color: #4897c4;
  color: #f1f1f1;
}
#T_1c633_row68_col0 {
  background-color: #045a8d;
  color: #f1f1f1;
}
#T_1c633_row68_col1 {
  background-color: #045b8f;
  color: #f1f1f1;
}
#T_1c633_row68_col2 {
  background-color: #2c89bd;
  color: #f1f1f1;
}
#T_1c633_row69_col0 {
  background-color: #a4bcda;
  color: #000000;
}
#T_1c633_row69_col1 {
  background-color: #a5bddb;
  color: #000000;
}
#T_1c633_row70_col0 {
  background-color: #8cb3d5;
  color: #000000;
}
#T_1c633_row71_col0 {
  background-color: #f3edf5;
  color: #000000;
}
#T_1c633_row71_col1 {
  background-color: #f4edf6;
  color: #000000;
}
#T_1c633_row71_col2 {
  background-color: #8bb2d4;
  color: #000000;
}
#T_1c633_row72_col0 {
  background-color: #e8e4f0;
  color: #000000;
}
#T_1c633_row72_col1 {
  background-color: #eae6f1;
  color: #000000;
}
#T_1c633_row73_col1 {
  background-color: #187cb6;
  color: #f1f1f1;
}
#T_1c633_row73_col2 {
  background-color: #4697c4;
  color: #f1f1f1;
}
#T_1c633_row74_col1 {
  background-color: #93b5d6;
  color: #000000;
}
#T_1c633_row74_col2 {
  background-color: #80aed2;
  color: #f1f1f1;
}
#T_1c633_row75_col0 {
  background-color: #c8cde4;
  color: #000000;
}
#T_1c633_row76_col1 {
  background-color: #d3d4e7;
  color: #000000;
}
#T_1c633_row77_col2 {
  background-color: #dddbec;
  color: #000000;
}
#T_1c633_row78_col0 {
  background-color: #cdd0e5;
  color: #000000;
}
#T_1c633_row78_col1, #T_1c633_row84_col0 {
  background-color: #ced0e6;
  color: #000000;
}
#T_1c633_row78_col2 {
  background-color: #9cb9d9;
  color: #000000;
}
#T_1c633_row79_col0 {
  background-color: #d0d1e6;
  color: #000000;
}
#T_1c633_row80_col2 {
  background-color: #e1dfed;
  color: #000000;
}
#T_1c633_row81_col0 {
  background-color: #045788;
  color: #f1f1f1;
}
#T_1c633_row81_col1 {
  background-color: #04588a;
  color: #f1f1f1;
}
#T_1c633_row81_col2 {
  background-color: #a7bddb;
  color: #000000;
}
#T_1c633_row82_col0 {
  background-color: #2786bb;
  color: #f1f1f1;
}
#T_1c633_row82_col1 {
  background-color: #2987bc;
  color: #f1f1f1;
}
#T_1c633_row85_col2 {
  background-color: #1077b4;
  color: #f1f1f1;
}
</style>
<table id="T_1c633">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_1c633_level0_col0" class="col_heading level0 col0" >MatGL Energy per Atom</th>
      <th id="T_1c633_level0_col1" class="col_heading level0 col1" >TF Energy per Atom</th>
      <th id="T_1c633_level0_col2" class="col_heading level0 col2" >Difference</th>
    </tr>
    <tr>
      <th class="index_name level0" >Formula</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_1c633_level0_row0" class="row_heading level0 row0" >Ac</th>
      <td id="T_1c633_row0_col0" class="data row0 col0" >-4.099037</td>
      <td id="T_1c633_row0_col1" class="data row0 col1" >-4.098995</td>
      <td id="T_1c633_row0_col2" class="data row0 col2" >-0.000042</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row1" class="row_heading level0 row1" >Ag</th>
      <td id="T_1c633_row1_col0" class="data row1 col0" >-2.811852</td>
      <td id="T_1c633_row1_col1" class="data row1 col1" >-2.798091</td>
      <td id="T_1c633_row1_col2" class="data row1 col2" >-0.013762</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row2" class="row_heading level0 row2" >Al</th>
      <td id="T_1c633_row2_col0" class="data row2 col0" >-3.742051</td>
      <td id="T_1c633_row2_col1" class="data row2 col1" >-3.759993</td>
      <td id="T_1c633_row2_col2" class="data row2 col2" >0.017942</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row3" class="row_heading level0 row3" >AlAs</th>
      <td id="T_1c633_row3_col0" class="data row3 col0" >-4.643467</td>
      <td id="T_1c633_row3_col1" class="data row3 col1" >-4.708796</td>
      <td id="T_1c633_row3_col2" class="data row3 col2" >0.065328</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row4" class="row_heading level0 row4" >AlP</th>
      <td id="T_1c633_row4_col0" class="data row4 col0" >-5.181847</td>
      <td id="T_1c633_row4_col1" class="data row4 col1" >-5.210092</td>
      <td id="T_1c633_row4_col2" class="data row4 col2" >0.028244</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row5" class="row_heading level0 row5" >AlSb</th>
      <td id="T_1c633_row5_col0" class="data row5 col0" >-4.093573</td>
      <td id="T_1c633_row5_col1" class="data row5 col1" >-4.078317</td>
      <td id="T_1c633_row5_col2" class="data row5 col2" >-0.015256</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row6" class="row_heading level0 row6" >Ar</th>
      <td id="T_1c633_row6_col0" class="data row6 col0" >-0.054958</td>
      <td id="T_1c633_row6_col1" class="data row6 col1" >-0.060761</td>
      <td id="T_1c633_row6_col2" class="data row6 col2" >0.005803</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row7" class="row_heading level0 row7" >Au</th>
      <td id="T_1c633_row7_col0" class="data row7 col0" >-3.264417</td>
      <td id="T_1c633_row7_col1" class="data row7 col1" >-3.265215</td>
      <td id="T_1c633_row7_col2" class="data row7 col2" >0.000799</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row8" class="row_heading level0 row8" >BN</th>
      <td id="T_1c633_row8_col0" class="data row8 col0" >-8.705638</td>
      <td id="T_1c633_row8_col1" class="data row8 col1" >-8.660206</td>
      <td id="T_1c633_row8_col2" class="data row8 col2" >-0.045432</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row9" class="row_heading level0 row9" >BP</th>
      <td id="T_1c633_row9_col0" class="data row9 col0" >-6.451569</td>
      <td id="T_1c633_row9_col1" class="data row9 col1" >-6.459996</td>
      <td id="T_1c633_row9_col2" class="data row9 col2" >0.008428</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row10" class="row_heading level0 row10" >C</th>
      <td id="T_1c633_row10_col0" class="data row10 col0" >-9.086294</td>
      <td id="T_1c633_row10_col1" class="data row10 col1" >-9.062232</td>
      <td id="T_1c633_row10_col2" class="data row10 col2" >-0.024062</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row11" class="row_heading level0 row11" >Ca</th>
      <td id="T_1c633_row11_col0" class="data row11 col0" >-1.989119</td>
      <td id="T_1c633_row11_col1" class="data row11 col1" >-1.992147</td>
      <td id="T_1c633_row11_col2" class="data row11 col2" >0.003029</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row12" class="row_heading level0 row12" >CaVO3</th>
      <td id="T_1c633_row12_col0" class="data row12 col0" >-7.302930</td>
      <td id="T_1c633_row12_col1" class="data row12 col1" >-7.287400</td>
      <td id="T_1c633_row12_col2" class="data row12 col2" >-0.015529</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row13" class="row_heading level0 row13" >CdS</th>
      <td id="T_1c633_row13_col0" class="data row13 col0" >-3.139587</td>
      <td id="T_1c633_row13_col1" class="data row13 col1" >-3.166539</td>
      <td id="T_1c633_row13_col2" class="data row13 col2" >0.026952</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row14" class="row_heading level0 row14" >CdSe</th>
      <td id="T_1c633_row14_col0" class="data row14 col0" >-2.823035</td>
      <td id="T_1c633_row14_col1" class="data row14 col1" >-2.808182</td>
      <td id="T_1c633_row14_col2" class="data row14 col2" >-0.014854</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row15" class="row_heading level0 row15" >CdTe</th>
      <td id="T_1c633_row15_col0" class="data row15 col0" >-2.479242</td>
      <td id="T_1c633_row15_col1" class="data row15 col1" >-2.479013</td>
      <td id="T_1c633_row15_col2" class="data row15 col2" >-0.000229</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row16" class="row_heading level0 row16" >Ce</th>
      <td id="T_1c633_row16_col0" class="data row16 col0" >-5.877059</td>
      <td id="T_1c633_row16_col1" class="data row16 col1" >-5.949479</td>
      <td id="T_1c633_row16_col2" class="data row16 col2" >0.072420</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row17" class="row_heading level0 row17" >Cr</th>
      <td id="T_1c633_row17_col0" class="data row17 col0" >-9.521768</td>
      <td id="T_1c633_row17_col1" class="data row17 col1" >-9.443845</td>
      <td id="T_1c633_row17_col2" class="data row17 col2" >-0.077923</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row18" class="row_heading level0 row18" >CrN</th>
      <td id="T_1c633_row18_col0" class="data row18 col0" >-9.318823</td>
      <td id="T_1c633_row18_col1" class="data row18 col1" >-9.262450</td>
      <td id="T_1c633_row18_col2" class="data row18 col2" >-0.056373</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row19" class="row_heading level0 row19" >CsCl</th>
      <td id="T_1c633_row19_col0" class="data row19 col0" >-3.254575</td>
      <td id="T_1c633_row19_col1" class="data row19 col1" >-3.266138</td>
      <td id="T_1c633_row19_col2" class="data row19 col2" >0.011563</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row20" class="row_heading level0 row20" >CsF</th>
      <td id="T_1c633_row20_col0" class="data row20 col0" >-4.006769</td>
      <td id="T_1c633_row20_col1" class="data row20 col1" >-4.002890</td>
      <td id="T_1c633_row20_col2" class="data row20 col2" >-0.003879</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row21" class="row_heading level0 row21" >CsI</th>
      <td id="T_1c633_row21_col0" class="data row21 col0" >-2.759116</td>
      <td id="T_1c633_row21_col1" class="data row21 col1" >-2.735941</td>
      <td id="T_1c633_row21_col2" class="data row21 col2" >-0.023175</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row22" class="row_heading level0 row22" >Cu</th>
      <td id="T_1c633_row22_col0" class="data row22 col0" >-4.085120</td>
      <td id="T_1c633_row22_col1" class="data row22 col1" >-4.062448</td>
      <td id="T_1c633_row22_col2" class="data row22 col2" >-0.022672</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row23" class="row_heading level0 row23" >Eu</th>
      <td id="T_1c633_row23_col0" class="data row23 col0" >-10.227932</td>
      <td id="T_1c633_row23_col1" class="data row23 col1" >-10.343983</td>
      <td id="T_1c633_row23_col2" class="data row23 col2" >0.116051</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row24" class="row_heading level0 row24" >EuTiO3</th>
      <td id="T_1c633_row24_col0" class="data row24 col0" >-9.730120</td>
      <td id="T_1c633_row24_col1" class="data row24 col1" >-9.730616</td>
      <td id="T_1c633_row24_col2" class="data row24 col2" >0.000496</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row25" class="row_heading level0 row25" >Fe</th>
      <td id="T_1c633_row25_col0" class="data row25 col0" >-8.468353</td>
      <td id="T_1c633_row25_col1" class="data row25 col1" >-8.425884</td>
      <td id="T_1c633_row25_col2" class="data row25 col2" >-0.042469</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row26" class="row_heading level0 row26" >GaAs</th>
      <td id="T_1c633_row26_col0" class="data row26 col0" >-4.171322</td>
      <td id="T_1c633_row26_col1" class="data row26 col1" >-4.160844</td>
      <td id="T_1c633_row26_col2" class="data row26 col2" >-0.010478</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row27" class="row_heading level0 row27" >GaP</th>
      <td id="T_1c633_row27_col0" class="data row27 col0" >-4.577314</td>
      <td id="T_1c633_row27_col1" class="data row27 col1" >-4.621902</td>
      <td id="T_1c633_row27_col2" class="data row27 col2" >0.044588</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row28" class="row_heading level0 row28" >GaSb</th>
      <td id="T_1c633_row28_col0" class="data row28 col0" >-3.728891</td>
      <td id="T_1c633_row28_col1" class="data row28 col1" >-3.710316</td>
      <td id="T_1c633_row28_col2" class="data row28 col2" >-0.018575</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row29" class="row_heading level0 row29" >Ge</th>
      <td id="T_1c633_row29_col0" class="data row29 col0" >-4.586413</td>
      <td id="T_1c633_row29_col1" class="data row29 col1" >-4.565431</td>
      <td id="T_1c633_row29_col2" class="data row29 col2" >-0.020982</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row30" class="row_heading level0 row30" >HfC</th>
      <td id="T_1c633_row30_col0" class="data row30 col0" >-10.528796</td>
      <td id="T_1c633_row30_col1" class="data row30 col1" >-10.526671</td>
      <td id="T_1c633_row30_col2" class="data row30 col2" >-0.002125</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row31" class="row_heading level0 row31" >HfN</th>
      <td id="T_1c633_row31_col0" class="data row31 col0" >-10.883002</td>
      <td id="T_1c633_row31_col1" class="data row31 col1" >-10.900499</td>
      <td id="T_1c633_row31_col2" class="data row31 col2" >0.017497</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row32" class="row_heading level0 row32" >InAs</th>
      <td id="T_1c633_row32_col0" class="data row32 col0" >-3.841607</td>
      <td id="T_1c633_row32_col1" class="data row32 col1" >-3.892487</td>
      <td id="T_1c633_row32_col2" class="data row32 col2" >0.050880</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row33" class="row_heading level0 row33" >InP</th>
      <td id="T_1c633_row33_col0" class="data row33 col0" >-4.277247</td>
      <td id="T_1c633_row33_col1" class="data row33 col1" >-4.294531</td>
      <td id="T_1c633_row33_col2" class="data row33 col2" >0.017284</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row34" class="row_heading level0 row34" >InSb</th>
      <td id="T_1c633_row34_col0" class="data row34 col0" >-3.537774</td>
      <td id="T_1c633_row34_col1" class="data row34 col1" >-3.520551</td>
      <td id="T_1c633_row34_col2" class="data row34 col2" >-0.017223</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row35" class="row_heading level0 row35" >Ir</th>
      <td id="T_1c633_row35_col0" class="data row35 col0" >-8.934845</td>
      <td id="T_1c633_row35_col1" class="data row35 col1" >-8.873297</td>
      <td id="T_1c633_row35_col2" class="data row35 col2" >-0.061548</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row36" class="row_heading level0 row36" >K</th>
      <td id="T_1c633_row36_col0" class="data row36 col0" >-1.091753</td>
      <td id="T_1c633_row36_col1" class="data row36 col1" >-1.095718</td>
      <td id="T_1c633_row36_col2" class="data row36 col2" >0.003964</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row37" class="row_heading level0 row37" >KBr</th>
      <td id="T_1c633_row37_col0" class="data row37 col0" >-3.131697</td>
      <td id="T_1c633_row37_col1" class="data row37 col1" >-3.137578</td>
      <td id="T_1c633_row37_col2" class="data row37 col2" >0.005881</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row38" class="row_heading level0 row38" >KCl</th>
      <td id="T_1c633_row38_col0" class="data row38 col0" >-3.430560</td>
      <td id="T_1c633_row38_col1" class="data row38 col1" >-3.429949</td>
      <td id="T_1c633_row38_col2" class="data row38 col2" >-0.000612</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row39" class="row_heading level0 row39" >KF</th>
      <td id="T_1c633_row39_col0" class="data row39 col0" >-4.202847</td>
      <td id="T_1c633_row39_col1" class="data row39 col1" >-4.192967</td>
      <td id="T_1c633_row39_col2" class="data row39 col2" >-0.009879</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row40" class="row_heading level0 row40" >KI</th>
      <td id="T_1c633_row40_col0" class="data row40 col0" >-2.798555</td>
      <td id="T_1c633_row40_col1" class="data row40 col1" >-2.776462</td>
      <td id="T_1c633_row40_col2" class="data row40 col2" >-0.022093</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row41" class="row_heading level0 row41" >KTaO3</th>
      <td id="T_1c633_row41_col0" class="data row41 col0" >-8.208177</td>
      <td id="T_1c633_row41_col1" class="data row41 col1" >-8.184317</td>
      <td id="T_1c633_row41_col2" class="data row41 col2" >-0.023861</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row42" class="row_heading level0 row42" >Li</th>
      <td id="T_1c633_row42_col0" class="data row42 col0" >-1.902391</td>
      <td id="T_1c633_row42_col1" class="data row42 col1" >-1.934193</td>
      <td id="T_1c633_row42_col2" class="data row42 col2" >0.031802</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row43" class="row_heading level0 row43" >LiBr</th>
      <td id="T_1c633_row43_col0" class="data row43 col0" >-3.315594</td>
      <td id="T_1c633_row43_col1" class="data row43 col1" >-3.306401</td>
      <td id="T_1c633_row43_col2" class="data row43 col2" >-0.009192</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row44" class="row_heading level0 row44" >LiCl</th>
      <td id="T_1c633_row44_col0" class="data row44 col0" >-3.680833</td>
      <td id="T_1c633_row44_col1" class="data row44 col1" >-3.690298</td>
      <td id="T_1c633_row44_col2" class="data row44 col2" >0.009465</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row45" class="row_heading level0 row45" >LiF</th>
      <td id="T_1c633_row45_col0" class="data row45 col0" >-4.837592</td>
      <td id="T_1c633_row45_col1" class="data row45 col1" >-4.833163</td>
      <td id="T_1c633_row45_col2" class="data row45 col2" >-0.004429</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row46" class="row_heading level0 row46" >LiI</th>
      <td id="T_1c633_row46_col0" class="data row46 col0" >-2.913825</td>
      <td id="T_1c633_row46_col1" class="data row46 col1" >-2.912904</td>
      <td id="T_1c633_row46_col2" class="data row46 col2" >-0.000921</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row47" class="row_heading level0 row47" >MgO</th>
      <td id="T_1c633_row47_col0" class="data row47 col0" >-5.967094</td>
      <td id="T_1c633_row47_col1" class="data row47 col1" >-5.929637</td>
      <td id="T_1c633_row47_col2" class="data row47 col2" >-0.037457</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row48" class="row_heading level0 row48" >Mo</th>
      <td id="T_1c633_row48_col0" class="data row48 col0" >-10.859205</td>
      <td id="T_1c633_row48_col1" class="data row48 col1" >-10.745793</td>
      <td id="T_1c633_row48_col2" class="data row48 col2" >-0.113412</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row49" class="row_heading level0 row49" >Na</th>
      <td id="T_1c633_row49_col0" class="data row49 col0" >-1.302357</td>
      <td id="T_1c633_row49_col1" class="data row49 col1" >-1.299608</td>
      <td id="T_1c633_row49_col2" class="data row49 col2" >-0.002749</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row50" class="row_heading level0 row50" >NaBr</th>
      <td id="T_1c633_row50_col0" class="data row50 col0" >-3.025174</td>
      <td id="T_1c633_row50_col1" class="data row50 col1" >-3.045708</td>
      <td id="T_1c633_row50_col2" class="data row50 col2" >0.020535</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row51" class="row_heading level0 row51" >NaCl</th>
      <td id="T_1c633_row51_col0" class="data row51 col0" >-3.382240</td>
      <td id="T_1c633_row51_col1" class="data row51 col1" >-3.371249</td>
      <td id="T_1c633_row51_col2" class="data row51 col2" >-0.010991</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row52" class="row_heading level0 row52" >NaF</th>
      <td id="T_1c633_row52_col0" class="data row52 col0" >-4.317245</td>
      <td id="T_1c633_row52_col1" class="data row52 col1" >-4.293275</td>
      <td id="T_1c633_row52_col2" class="data row52 col2" >-0.023970</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row53" class="row_heading level0 row53" >NaI</th>
      <td id="T_1c633_row53_col0" class="data row53 col0" >-2.688805</td>
      <td id="T_1c633_row53_col1" class="data row53 col1" >-2.685169</td>
      <td id="T_1c633_row53_col2" class="data row53 col2" >-0.003637</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row54" class="row_heading level0 row54" >Nb</th>
      <td id="T_1c633_row54_col0" class="data row54 col0" >-10.054744</td>
      <td id="T_1c633_row54_col1" class="data row54 col1" >-10.031998</td>
      <td id="T_1c633_row54_col2" class="data row54 col2" >-0.022746</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row55" class="row_heading level0 row55" >NbN</th>
      <td id="T_1c633_row55_col0" class="data row55 col0" >-10.087329</td>
      <td id="T_1c633_row55_col1" class="data row55 col1" >-10.096134</td>
      <td id="T_1c633_row55_col2" class="data row55 col2" >0.008805</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row56" class="row_heading level0 row56" >Ne</th>
      <td id="T_1c633_row56_col0" class="data row56 col0" >-0.029663</td>
      <td id="T_1c633_row56_col1" class="data row56 col1" >0.021365</td>
      <td id="T_1c633_row56_col2" class="data row56 col2" >-0.051028</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row57" class="row_heading level0 row57" >Ni</th>
      <td id="T_1c633_row57_col0" class="data row57 col0" >-5.761651</td>
      <td id="T_1c633_row57_col1" class="data row57 col1" >-5.741039</td>
      <td id="T_1c633_row57_col2" class="data row57 col2" >-0.020611</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row58" class="row_heading level0 row58" >Pb</th>
      <td id="T_1c633_row58_col0" class="data row58 col0" >-3.697291</td>
      <td id="T_1c633_row58_col1" class="data row58 col1" >-3.713206</td>
      <td id="T_1c633_row58_col2" class="data row58 col2" >0.015915</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row59" class="row_heading level0 row59" >Pd</th>
      <td id="T_1c633_row59_col0" class="data row59 col0" >-5.178584</td>
      <td id="T_1c633_row59_col1" class="data row59 col1" >-5.119916</td>
      <td id="T_1c633_row59_col2" class="data row59 col2" >-0.058668</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row60" class="row_heading level0 row60" >Pt</th>
      <td id="T_1c633_row60_col0" class="data row60 col0" >-6.060041</td>
      <td id="T_1c633_row60_col1" class="data row60 col1" >-6.083516</td>
      <td id="T_1c633_row60_col2" class="data row60 col2" >0.023475</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row61" class="row_heading level0 row61" >RbBr</th>
      <td id="T_1c633_row61_col0" class="data row61 col0" >-3.067284</td>
      <td id="T_1c633_row61_col1" class="data row61 col1" >-3.065766</td>
      <td id="T_1c633_row61_col2" class="data row61 col2" >-0.001518</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row62" class="row_heading level0 row62" >RbCl</th>
      <td id="T_1c633_row62_col0" class="data row62 col0" >-3.350589</td>
      <td id="T_1c633_row62_col1" class="data row62 col1" >-3.344019</td>
      <td id="T_1c633_row62_col2" class="data row62 col2" >-0.006570</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row63" class="row_heading level0 row63" >RbF</th>
      <td id="T_1c633_row63_col0" class="data row63 col0" >-4.067951</td>
      <td id="T_1c633_row63_col1" class="data row63 col1" >-4.058086</td>
      <td id="T_1c633_row63_col2" class="data row63 col2" >-0.009864</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row64" class="row_heading level0 row64" >RbI</th>
      <td id="T_1c633_row64_col0" class="data row64 col0" >-2.745140</td>
      <td id="T_1c633_row64_col1" class="data row64 col1" >-2.714488</td>
      <td id="T_1c633_row64_col2" class="data row64 col2" >-0.030651</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row65" class="row_heading level0 row65" >Rh</th>
      <td id="T_1c633_row65_col0" class="data row65 col0" >-7.377921</td>
      <td id="T_1c633_row65_col1" class="data row65 col1" >-7.305958</td>
      <td id="T_1c633_row65_col2" class="data row65 col2" >-0.071963</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row66" class="row_heading level0 row66" >ScN</th>
      <td id="T_1c633_row66_col0" class="data row66 col0" >-9.249602</td>
      <td id="T_1c633_row66_col1" class="data row66 col1" >-9.248843</td>
      <td id="T_1c633_row66_col2" class="data row66 col2" >-0.000759</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row67" class="row_heading level0 row67" >Si</th>
      <td id="T_1c633_row67_col0" class="data row67 col0" >-5.358987</td>
      <td id="T_1c633_row67_col1" class="data row67 col1" >-5.355485</td>
      <td id="T_1c633_row67_col2" class="data row67 col2" >-0.003502</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row68" class="row_heading level0 row68" >Sr</th>
      <td id="T_1c633_row68_col0" class="data row68 col0" >-1.685335</td>
      <td id="T_1c633_row68_col1" class="data row68 col1" >-1.721167</td>
      <td id="T_1c633_row68_col2" class="data row68 col2" >0.035832</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row69" class="row_heading level0 row69" >SrTiO3</th>
      <td id="T_1c633_row69_col0" class="data row69 col0" >-8.016813</td>
      <td id="T_1c633_row69_col1" class="data row69 col1" >-8.012626</td>
      <td id="T_1c633_row69_col2" class="data row69 col2" >-0.004187</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row70" class="row_heading level0 row70" >SrVO3</th>
      <td id="T_1c633_row70_col0" class="data row70 col0" >-7.254021</td>
      <td id="T_1c633_row70_col1" class="data row70 col1" >-7.254895</td>
      <td id="T_1c633_row70_col2" class="data row70 col2" >0.000874</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row71" class="row_heading level0 row71" >Ta</th>
      <td id="T_1c633_row71_col0" class="data row71 col0" >-11.892914</td>
      <td id="T_1c633_row71_col1" class="data row71 col1" >-11.881514</td>
      <td id="T_1c633_row71_col2" class="data row71 col2" >-0.011400</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row72" class="row_heading level0 row72" >TaC</th>
      <td id="T_1c633_row72_col0" class="data row72 col0" >-11.098632</td>
      <td id="T_1c633_row72_col1" class="data row72 col1" >-11.138849</td>
      <td id="T_1c633_row72_col2" class="data row72 col2" >0.040217</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row73" class="row_heading level0 row73" >TePb</th>
      <td id="T_1c633_row73_col0" class="data row73 col0" >-3.826451</td>
      <td id="T_1c633_row73_col1" class="data row73 col1" >-3.849249</td>
      <td id="T_1c633_row73_col2" class="data row73 col2" >0.022798</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row74" class="row_heading level0 row74" >Th</th>
      <td id="T_1c633_row74_col0" class="data row74 col0" >-7.416744</td>
      <td id="T_1c633_row74_col1" class="data row74 col1" >-7.411212</td>
      <td id="T_1c633_row74_col2" class="data row74 col2" >-0.005532</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row75" class="row_heading level0 row75" >TiC</th>
      <td id="T_1c633_row75_col0" class="data row75 col0" >-9.375763</td>
      <td id="T_1c633_row75_col1" class="data row75 col1" >-9.409308</td>
      <td id="T_1c633_row75_col2" class="data row75 col2" >0.033545</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row76" class="row_heading level0 row76" >TiN</th>
      <td id="T_1c633_row76_col0" class="data row76 col0" >-9.808543</td>
      <td id="T_1c633_row76_col1" class="data row76 col1" >-9.826851</td>
      <td id="T_1c633_row76_col2" class="data row76 col2" >0.018308</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row77" class="row_heading level0 row77" >V</th>
      <td id="T_1c633_row77_col0" class="data row77 col0" >-9.127023</td>
      <td id="T_1c633_row77_col1" class="data row77 col1" >-9.058248</td>
      <td id="T_1c633_row77_col2" class="data row77 col2" >-0.068775</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row78" class="row_heading level0 row78" >VC</th>
      <td id="T_1c633_row78_col0" class="data row78 col0" >-9.574696</td>
      <td id="T_1c633_row78_col1" class="data row78 col1" >-9.552849</td>
      <td id="T_1c633_row78_col2" class="data row78 col2" >-0.021847</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row79" class="row_heading level0 row79" >VN</th>
      <td id="T_1c633_row79_col0" class="data row79 col0" >-9.676881</td>
      <td id="T_1c633_row79_col1" class="data row79 col1" >-9.667151</td>
      <td id="T_1c633_row79_col2" class="data row79 col2" >-0.009729</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row80" class="row_heading level0 row80" >W</th>
      <td id="T_1c633_row80_col0" class="data row80 col0" >-12.944801</td>
      <td id="T_1c633_row80_col1" class="data row80 col1" >-12.871328</td>
      <td id="T_1c633_row80_col2" class="data row80 col2" >-0.073473</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row81" class="row_heading level0 row81" >Yb</th>
      <td id="T_1c633_row81_col0" class="data row81 col0" >-1.524300</td>
      <td id="T_1c633_row81_col1" class="data row81 col1" >-1.496681</td>
      <td id="T_1c633_row81_col2" class="data row81 col2" >-0.027619</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row82" class="row_heading level0 row82" >ZnO</th>
      <td id="T_1c633_row82_col0" class="data row82 col0" >-4.404037</td>
      <td id="T_1c633_row82_col1" class="data row82 col1" >-4.390760</td>
      <td id="T_1c633_row82_col2" class="data row82 col2" >-0.013277</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row83" class="row_heading level0 row83" >ZnS</th>
      <td id="T_1c633_row83_col0" class="data row83 col0" >-3.501024</td>
      <td id="T_1c633_row83_col1" class="data row83 col1" >-3.497520</td>
      <td id="T_1c633_row83_col2" class="data row83 col2" >-0.003503</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row84" class="row_heading level0 row84" >ZrC</th>
      <td id="T_1c633_row84_col0" class="data row84 col0" >-9.653810</td>
      <td id="T_1c633_row84_col1" class="data row84 col1" >-9.696436</td>
      <td id="T_1c633_row84_col2" class="data row84 col2" >0.042626</td>
    </tr>
    <tr>
      <th id="T_1c633_level0_row85" class="row_heading level0 row85" >ZrN</th>
      <td id="T_1c633_row85_col0" class="data row85 col0" >-10.019135</td>
      <td id="T_1c633_row85_col1" class="data row85 col1" >-10.071356</td>
      <td id="T_1c633_row85_col2" class="data row85 col2" >0.052220</td>
    </tr>
  </tbody>
</table>





```python
_ = df.plot(x="MatGL Energy per Atom", y="TF Energy per Atom", kind="scatter")
```



![png](assets/Validation%20of%20Refitted%20MatGL%20versus%20Original%20TF%20M3GNet_8_0.png)




```python
mae = df["Difference"].abs().mean()
print(f"The mean absolute difference between the MatGL and TF energy per atom is {mae:.5f} eV/atom.")
```

    The mean absolute difference between the MatGL and TF energy per atom is 0.02466 eV/atom.
