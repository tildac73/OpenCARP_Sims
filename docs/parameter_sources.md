# Parameter sources for the reference model (Step 3, readiness plan)

Status: Sourced = value read from the primary paper; Open = no primary value retrieved yet.
Only papers whose text was actually read are cited. Values are as reported; units converted where noted.
Last updated 28 Oct 2026 (CV research).

| Quantity | Current model (v2-pilot) | Sourced value | Source | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| LA conduction velocity, sinus rhythm, normal-voltage tissue | 0.35 m/s (planar, along fibres) | 1.25 +/- 0.38 m/s (patients without fibrotic cardiomyopathy, n = 25); anterior 1.11 +/- 0.34, posterior 1.38 +/- 0.37 m/s | Heil et al. 2023, doi:10.1007/s00380-023-02282-5 (Table 2) | Sourced | Persistent-AF cohort mapped in sinus rhythm (>1000 points). EAM CV mixes wavefront directions, so it is not a pure longitudinal CV. Text and table differ for some values; Table 2 used. The model's 0.35 m/s is about 3.5x slower: must be resolved in Step 3. |
| Human atrial CV, sinus rhythm, epicardial high-density (direction-mixed) | - | LA median 89.6 cm/s; overall median 90.0 (62.4-116.8) cm/s; 412 patients | van Schie et al. 2021, Europace 23:1815-1825, doi:10.1093/europace/euab088 | Sourced | Intra-operative epicardial arrays (2 mm spacing). Largest cohort found. |
| Human atrial CV, sinus rhythm, biatrial (direction-mixed) | - | 90 +/- 8 (AF) vs 92 +/- 6 cm/s (no AF) | Heida et al. 2021, J Clin Med 10:2614, doi:10.3390/jcm10122614 | Sourced | Same group; epicardial. Also reports (secondary, not read) Zheng et al.: endocardial control LA 70 +/- 10, RA 83 +/- 13 cm/s. |
| Human RA free-wall CV by pacing direction | - | SR 88 +/- 9 cm/s; paced 0/90/180/270 deg: 81 +/- 16, 74 +/- 14, 79 +/- 12, 78 +/- 20 cm/s; no significant direction dependence | Hansson et al. 1998, Eur Heart J 19:293-300, doi:10.1053/euhj.1997.0742 (abstract) | Sourced | Directions relative to the AV groove, not measured fibre direction; 12 patients; not a longitudinal/transverse measurement. |
| Longitudinal / transverse CV and anisotropy ratio, human atrial body | gi 0.174 / 0.094 S/m (ratio 1.85) | No numerical human atrial-body value found. Reviews: anisotropy ratio ~10 in crista terminalis, ~2 in ventricle; atrial-body ratio lower than crista terminalis and Bachmann's bundle | Valderrabano 2007, Prog Biophys Mol Biol 94:144-168; Kotadia et al. 2020, Arrhythm Electrophysiol Rev 9(4):202-10, doi:10.15420/aer.2020.04 | Open (bounded) | Primary candidates cited by Kotadia: Maesen et al. (Circ Arrhythm Electrophysiol, goat AF model); Morton et al. 2003 Circulation (human RA, atrial septal defect). Roney et al. 2019 clinical results still unread. |
| Longitudinal / transverse CV, canine ventricle (reference only) | - | 57 +/- 6 / 24 +/- 3 cm/s (ratio 2.4) | Roberts and Scher 1982, Circ Res 50:342-351 | Sourced | Canine LV epicardium in situ; not atrial. |
| Intracellular / extracellular conductivities | gi 0.174 / 0.094, ge 0.625 / 0.338 S/m (source unrecorded) | Resistivities (ohm cm): r_il 291 +/- 38, r_it 1677 +/- 331, r_el 852 +/- 232, r_et 1247 +/- 210, i.e. sigma_il 0.344, sigma_it 0.060, sigma_el 0.117, sigma_et 0.080 S/m; gross tissue 213 / 705 ohm cm | Roberts and Scher 1982, Circ Res 50:342-351 | Sourced (canine ventricle) | Best-fit bidomain values for canine LV in situ; no atrial-specific bidomain measurements found. Intracellular anisotropy 5.8, extracellular 1.5 (unequal). Current model values differ in both magnitude and ratio. Clerc 1976 still not read. |
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

## Conductivity strategy (decided 28 Oct, R4: DL-2026-10-28-04)
No atrial bidomain conductivities were found. Common practice is to take the anisotropy ratios from a measured set (here Roberts and Scher 1982) and scale magnitudes to reach a sourced CV at the production mesh (openCARP CV tuning). This keeps unequal anisotropy, which matters for extracellular fields. The scaling must be recorded as a calibration, not a measurement.

## Access needed
Scite access lapsed and several primary papers are paywalled. Received 28 Oct: Roberts and Scher 1982; Kotadia et al. 2020; Valderrabano 2007. Still needed: a primary human atrial-body anisotropy measurement (e.g. Morton et al. 2003), Clerc 1976.

## Conduction velocity target (decided 28 Oct: DL-2026-10-28-03)
No direction-resolved (fibre-aligned) human atrial-body CV was found. Clinical sinus-rhythm CVs are local
estimates that mix wavefront directions relative to fibres; four independent sources agree on ~0.9 m/s
(range of reported central values 0.70-1.25 m/s). Target: **planar CV along fibres = 0.90 m/s** (van Schie
et al. 2021 LA median, largest cohort), transverse CV set by the Roberts and Scher anisotropy ratios.
Because a direction-mixed median lies between transverse and longitudinal CV, 0.90 m/s along fibres is a
conservative (low) choice; sensitivity at 0.70 and 1.25 m/s brackets the reported central values.

## Expected consequence of the CV target for conductivity magnitudes (flag for Step 6)
Estimate only, from the pilot (0.352 m/s at monodomain sigma_l 0.136 S/m, 0.5 mm; CV ~ sqrt(sigma)): reaching
0.90 m/s needs a monodomain sigma_l of about 0.9 S/m, i.e. about 10x the Roberts and Scher magnitudes
(sigma_el ~1.2 S/m). The CV stage will give the actual factor. Two points follow:
1. An extracellular longitudinal conductivity above blood (0.74 S/m) is physically implausible. In the unbounded homogeneous
   medium, at a given CV, the magnitudes only scale the electrogram amplitude (no calibration since FM-25: amplitudes are reported in model units);
   once a bath is added (Step 6) the tissue-to-blood conductivity ratio changes the field itself.
2. CV depends on sigma / (beta * Cm). The same CV could instead be reached by keeping measured conductivities
   and lowering the surface-to-volume ratio beta (openCARP default 0.14 um^-1). Which to adjust is a
   modelling decision (extends R4); it needs a sourced beta for human atrium before Step 6.

## Added 3 Nov 2026 (final model v4, FM-08 to FM-10)
| Quantity | Model value | Sourced value | Source | Status |
| --- | --- | --- | --- | --- |
| LA wall thickness by region, CT | inferior 1.1 mm, roof 1.6 mm | median inferior posterior wall 1.14 mm (5-95 %: 0.72-2.71); roof 1.55 mm (0.70-3.72); superior posterior wall 1.73 mm; 193 subjects | Whitaker et al., Int J Cardiol Heart Vasc 2021;32:100694, doi:10.1016/j.ijcha.2020.100694 | Sourced (read in full). Rounded to the 0.1 mm transmural mesh. Contradicts the regional argument in Wang's discussion (inferior thicker, post-mortem source). |
| EnSite X acquisition rate | 2 kHz | 2000 Hz sampling, nominal 300 Hz digital low-pass, OT Near Field PF: wavelet-derived, PP >= 0.04 mV, voltage +-30 ms of the PF annotation | Europace 2024;26:euae154 (VT study using EnSite X) | Sourced for that study; Wang's own sampling rate not reported [VERIFY]. |
| Bipolar/omnipolar filter | 30-500 Hz | 30-500 Hz | Wang supplement | Sourced. |
| OT Near Field wavelet, scales, gate | Morlet omega0 = 6, 30-500 Hz, 10 % gate | proprietary | Abbott | Assumption. |
| Near-field cut-off | 4 mm | 4 mm (human atrium) | Schlageter et al. 2024, as cited in Wang's manuscript | Sourced (secondary). |

## Added 3 Nov 2026 (revision FM-23 to FM-30)
| Quantity | Model value | Sourced value | Source | Status |
| --- | --- | --- | --- | --- |
| Ablation lesion width | 5.6 mm strip | 5.6 +- 1.2 mm | Wang supplement, citing Friedman et al. 2021 (not read directly) | decided, [VERIFY] primary source |
| Clinical bipolar voltage | not applied | 1.44 mV inferior, 0.97 mV roof (healthy, median) | Wang Table 2 | reference only (FM-25) |
| Near/far lateral cut-off | 4 mm | none | one HD Grid electrode spacing; Schlageter et al. 2024's 4 mm is electrode-to-tissue distance | convention (FM-30) |
| Forward model | monodomain Vm, sigma_i source | monodomain + forward model: electrograms indistinguishable from bidomain | Potse et al. 2006, IEEE TBME 53(12):2425-2435 (abstract read; bibliographic record checked 4 Nov 2026) | decided (FM-23) |
| Electrode model | equipotential conductor (openCARP Laplace, electrode at 1 mV, z = u / I) | HD Grid electrodes are metal; against passive averaging: lead field 0.07 %, recorded amplitudes about 1 % (test slab) | VL-2026-11-07-01, -03 | decided (FM-34) |

