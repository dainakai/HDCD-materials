# Experimental physical comparisons

Each of the 14 recordings contains 10,917 frames at 4,000 frames/s. The nominal observation volume is 10.24 × 10.24 × 102.3 mm³ = 10.72693248 cm³. Exposure is volume multiplied by the full recording duration. The pooled exposure is 0.00040987072659456 m³ s.

The gravitational prediction uses 5-µm diameter bins from 25 to 300 µm, the measured number density in each recording, terminal-velocity differences, and the bundled Davis–Jonas–Hall collision-efficiency table. Each unordered fine-bin pair contributes once. Distinct fine diameter bins within the same 25-µm display bin still contribute to its diagonal cell. The prediction is integrated over each recording's exposure, then summed across recordings.

Uncertainty bars show the conditional Poisson standard deviation, `sqrt(expected count)`. They do not include uncertainty in measured density, collision efficiency or unreviewed detections. The sum is 222.410280 expected collisions with a standard deviation of 14.913426. The 186 confirmed events give an observed/predicted ratio of 0.836292 and an observed lower-bound rate of 453,801.6207 m⁻³ s⁻¹.

Diameter consistency compares the measured daughter diameter with `(D1³ + D2³)^(1/3)`. Vertical-velocity consistency compares the measured post-event velocity with `(D1³ v1 + D2³ v2)/(D1³ + D2³)`. The analysis uses the observed diameter windows and median frame-to-frame displacement velocities. All three velocity windows require at least three observed points.

Of 186 confirmed collisions, 185 have both parent diameters for the pair map, 184 have usable parent and daughter diameters for the diameter diagnostic, and 179 have sufficient velocity observations. S0015 rank 34 lacks one parent diameter; S0022 rank 27 lacks a daughter diameter but retains both parents. These events remain in the total count. The feature-table minimum and maximum parent diameters are not used to fabricate a pair when one parent is missing.

The confirmed-event diameter RMSE is 19.8994 µm and the velocity RMSE is 0.0793867 m/s. A large velocity residual contributes 64.3% of squared error and has three post-event observations. Requiring five observations in each window leaves 173 confirmed events with velocity RMSE 0.0461580 m/s; this sensitivity calculation does not replace the primary three-point criterion.

The observation count is a lower bound because only the highest-ranked 80 candidates in each recording were reviewed. Review coverage uses the confirmed top-80 events as its denominator and is not absolute recall over the entire experimental recording.
