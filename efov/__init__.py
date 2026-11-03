"""Final eFOV slab model: openCARP simulation and the end-to-end analysis.

Modules, in the order the pipeline uses them:
    config        the configuration (config/final_model.json): load, check, derived values, fingerprints
    simulation    bench conditioning and the openCARP runs (opencarp/slab_experiment.py, carputils)
    physiology    activation, stability and CV checks; the measurement window
    forward       independent extracellular finite-element model (check and ground truth)
    electrograms  unipolar phi_e at every electrode; forward-model check
    recordings    unipoles, selected bipoles, omnipoles
    measurements  calibration, noise, peak-to-peak, profile checks
    changepoints  clinical-style decay-to-plateau fits
    groundtruth   exact contributing-tissue fields
    report        acceptance, report.md, figures
    pipeline      the stages in order (Run)
"""
