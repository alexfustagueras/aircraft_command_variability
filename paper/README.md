# Paper figures and numbers

Scripts behind the figures and the text numbers of the OSN26 paper. The figures
are in `figures/`; tables and statistics go to `diagnostics/runs/paper/`.
Every script runs from the repository root with its defaults: inference run
`inference_nodefdm_034_a320_4237`, synthetic pool run
`rq2_trajectory_pool_500_014_a320_4237` and library
`empirical_libraries_rq2_a320_4237_v8` (paths in `style.py`).

| Paper | Script | Output |
|---|---|---|
| Figure 1 | `h_sel_tau_example.py` | `fig_h_sel_tau_example.png` |
| Figure 2 | `speed_schedule.py` | `fig_speed_schedule.png` |
| Figure 3 | `event_anatomy.py` | `conditional_sampling_event_anatomy.png` |
| Figure 4, paths no flight flew | `target_path_walk.py --uniqueness` | `target_path_walk.png`, `target_path_uniqueness.csv` |
| Figure 5 | `rq1_replay_example.py` | `discordant_fidelity_example.png` |
| Figure 6, nearest-neighbour distances (appendix), Tables 2 and 3 | `rq2_pool_fidelity.py` | `rq2_bands_all_routes_altitude.png`, `rq2_nn_cdf_all_routes.png`, `rq2/*.csv` |
| Inferred targets and BDS 4,0 (appendix) | `selected_altitude_crosscheck.py` | `selected_altitude_crosscheck.png` |
| RDP tolerance (appendix) | `rdp_tolerance.py <sweep dir>` | `rdp_tolerance_complexity_fidelity_representation.png` |
| Individual profiles (appendix) | `rq2_individual_profiles.py` | `rq2_individual_profiles.png` |
| Text numbers | `paper_statistics.py` | `paper_statistics.json` |

```bash
.venv/bin/python paper/rq2_pool_fidelity.py
.venv/bin/python diagnostics/bin/rdp_epsilon_representation_sweep.py --output-dir diagnostics/runs/rdp_sweep
.venv/bin/python paper/rdp_tolerance.py diagnostics/runs/rdp_sweep
```