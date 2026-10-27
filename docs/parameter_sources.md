# Parameter sources for the reference model (Step 3, readiness plan)

Status: Sourced = value read from the primary paper; Open = no primary value retrieved yet.
Only papers whose text was actually read are cited. Values are as reported; units converted where noted.
Last updated 27 Oct 2026.

| Quantity | Current model (v2-pilot) | Sourced value | Source | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| LA conduction velocity, sinus rhythm, normal-voltage tissue | 0.35 m/s (planar, along fibres) | 1.25 +/- 0.38 m/s (patients without fibrotic cardiomyopathy, n = 25); anterior 1.11 +/- 0.34, posterior 1.38 +/- 0.37 m/s | Heil et al. 2023, doi:10.1007/s00380-023-02282-5 (Table 2) | Sourced | Persistent-AF cohort mapped in sinus rhythm (>1000 points). EAM CV mixes wavefront directions, so it is not a pure longitudinal CV. Text and table differ for some values; Table 2 used. The model's 0.35 m/s is about 3.5x slower: must be resolved in Step 3. |
| Longitudinal / transverse CV and anisotropy ratio, human atria | gi 0.174 / 0.094 S/m (ratio 1.85) | - | Candidates: Roney et al. 2019, doi:10.1016/j.compbiomed.2018.10.019 (clinical results section not yet read; simulation inputs 1.10 / 0.55 m/s are NOT human data) | Open | Needs a primary human measurement. |
| Intracellular / extracellular conductivities | gi 0.174 / 0.094, ge 0.625 / 0.338 S/m | - | Candidates: Clerc 1976 (J Physiol 255:335); Roberts and Scher 1982 | Open | Primary papers not retrievable here (not open access). Current values' source is unrecorded. |
| Blood conductivity | not modelled (unbounded homogeneous medium) | resistivity 135 ohm cm in vivo, normal haematocrit (= 0.74 S/m) | Traugott, Quail and White 1981, doi:10.1007/BF02442767 (abstract) | Sourced | Man, dog, rabbit; man Hct 31-48%. Temperature not stated in abstract. |
| Atrial wall thickness, ex vivo | 4 mm | mean 2.73 mm (5-95%: 0.98-4.38 mm), 8 human atria | Pashakhanloo et al. 2016, doi:10.1161/CIRCEP.116.004133 | Sourced | Sub-millimetre DT-MRI; walls < 0.5 mm not resolvable. |
| LA posterior wall thickness, CT | 4 mm | 1.48 +/- 0.21 mm, 61 AF patients | Li et al. 2024, doi:10.1186/s13019-024-02658-2 | Sourced | Non-gated 64-slice CT; resolution limits thin-wall accuracy. |
| Transmural fibre architecture | uniform, along x | Bilayer with nearly perpendicular layers and an abrupt mid-wall transition in posterior-inferior and anterior LA; constant through the wall on the lateral wall and roof | Pashakhanloo et al. 2016 (as above) | Sourced (qualitative) | No layer angles given numerically. Supports uniform fibres as defensible for lateral/roof, with a bilayer option for posterior/anterior (decision R11). |
| Outer (epicardial) medium conductivity | - | - | - | Open | Needed for Step 6. |
| Chronic-AF CRN parameter set | healthy CRN | - | - | Open | Needed for Step 9. |

## Implications already visible
1. Conduction velocity is the most urgent discrepancy: 0.35 m/s against ~1.25 m/s clinical sinus-rhythm CV. Step 3 must establish how much is mesh under-resolution and how much is conductivity choice, then tune to a sourced target.
2. The 4 mm wall sits at the upper end of reported values (ex vivo 0.98-4.38 mm; CT posterior wall ~1.5 mm). A thinner reference wall needs more elements through it (Step 3/7).
3. Fibre architecture is regionally heterogeneous; the reference choice (R11) should name the atrial region it represents.

## Access needed
Scite access lapsed and several primary papers are paywalled. Library PDFs of Clerc 1976, Roberts and Scher 1982, and a primary human atrial anisotropy study would close the open rows.
