"""Final eFOV slab model (v4-reference): openCARP simulation and the end-to-end analysis.

Modules, in the order the pipeline uses them:
    parfile        openCARP parameter files: read model.par, write each run's resolved file
    config         config/model.json + model.par: load, check, derived values, fingerprints
    fibres         uniform / bilayer / linear-rotation fibre fields
    meshbuild      run mesh (epicardial bath, fibres) and its node sets (stimulus, ground, phi_e box)
    simulation     bench conditioning and the openCARP runs (opencarp/slab_experiment.py, carputils)
    tune           beta for the CV target on a strip at the production mesh
    physiology     activation, planarity, direction and stability checks; the measurement window
    geometry       catheter layout, electrodes as mesh node sets
    electrograms   unipolar phi_e at every electrode
    recordings     unipoles, orientation-selected bipoles, omnipoles
    peakfrequency  EnSite OT Near Field approximation (Morlet wavelet)
    measurements   acquisition chain, calibration, noise, PP, peak frequency, profile checks
    export         Wang-format datasets for R/wang_replication.R
    leadfield      openCARP Laplace lead fields (Python fallback) and contribution coefficients
    groundtruth    exact contributing-tissue fields, classes, profiles, shape descriptors
    groundtruth_figures, report   figures, acceptance, report.md
    pipeline       the stages in order (Run); sweep: the parameter sweeps
"""
