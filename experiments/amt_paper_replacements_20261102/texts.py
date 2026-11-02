"""Publication wording. Values are inserted from the completed, hashed evaluation."""

FIGURE_LABELS = {
    7: 'fig:synthetic_hologram_distribution',
    8: 'fig:synthetic_hologram_validation_results',
    9: 'fig:image_feature_case_study',
}

CAPTIONS = {
    7: r'''Performance of holographic droplet detection and tracking using HoloD3 \citep{HoloD3} on the same synthetic test sequence used for the collision-detection evaluation in Fig.~\ref{fig:synthetic_hologram_validation_results} (seed 260745; 6,000 frames).
The particle measurements and trajectories are from the same reconstruction and tracking pipeline used as inputs to that evaluation.
(a) Particle-level precision--recall curve obtained by varying the particle-detection confidence threshold.
Histograms show (b) the in-plane position error, \(\sqrt{\Delta x^2+\Delta y^2}\), (c) the absolute axial-position error, \(|\Delta z|\), and (d) the signed relative diameter error, \(100(D_{\mathrm{meas}}-D_{\mathrm{true}})/D_{\mathrm{true}}\), for matched observations at the tracking-input confidence threshold of 0.1.
The errors in (b--d) use the smoothed trajectories supplied to collision detection; associations are determined in the image plane.
Annotations in (b,c) report the 50th, 90th, and 95th percentiles; those in (d) report the median and central 90\% and 95\% intervals.
(e) Ground-truth and measured number-density distributions.
(f) Cumulative numbers of trajectories with length greater than or equal to \(L\), for the ground-truth and reconstructed trajectories.''',
    8: r'''Event-detection performance and review burden of the proposed method, the depth-free variant, and MinIP-based 2D on the same synthetic test sequence as Fig.~\ref{fig:synthetic_hologram_distribution}, containing 2,000 ground-truth events.
The depth-free variant retains the measured trajectories and diameters used by the proposed method but excludes depth information during collision detection, whereas MinIP-based 2D independently estimates diameters and constructs two-dimensional trajectories from MinIP images before depth-free collision detection (Table~\ref{tab:method_comparison}).
(a) Event-level precision--recall curves; AP is the integral of the monotonically enveloped precision--recall curve.
(b) Precision (solid lines) and recall (dashed lines) among ranked candidates; tied scores are included together.
The vertical line marks the number of ground-truth events, \(N=2{,}000\).
(c) Number of false-positive candidates at each target recall, using the highest score threshold that attains or exceeds that recall.
(d) False-positive composition at a target recall of 0.8, separated into duplicate or competing detections and candidates with no ground-truth event within the matching gate.
The evaluation uses maximum-cardinality one-to-one matching within \(\pm5\) frames and an in-plane distance of \ensuremath{350\,\unit{\mu m}}, without an axial gate.''',
}


def figure9_caption(n):
    true, artifact = n['figure9_cases']
    return rf'''Illustrative examples of the effect of image features on candidate scores on the synthetic test sequence.
Reconstructed images are shown at \(-1.25\), \(0\), and \(+1.00\,\unit{{ms}}\) relative to the candidate time for (a) a true collision--coalescence event and (b) a shared-particle artifact, in which two reconstructed track IDs represent the same physical particle.
Red circles and blue squares mark actual observations of the terminating and continuing tracks, respectively; lines show their recent trajectories.
Each \(128\times128\)-pixel crop (\(1.28\times1.28\,\unit{{mm}}\)) follows the continuing track and is reconstructed at its estimated depth.
For display, each crop is linearly normalized between its first and 99th intensity percentiles; the scale bar is \ensuremath{{500\,\unit{{\mu m}}}}.
The diameters below (a) are ground-truth parent and daughter diameters.
(c) Candidate scores from the trajectory classifier and the image-feature ensemble, before relation-attention adjustment.
The true-event score changes from {true['trajectory_score']:.3f} to {true['image_feature_score']:.3f}, whereas the artifact score decreases from {artifact['trajectory_score']:.3f} to {artifact['image_feature_score']:.3f}.
The retained/rejected labels refer to the final candidate-selection policy.
These examples were selected for visibility after examining the results and illustrate individual candidates rather than an average effect over the test sequence.'''

TABLE4_CAPTION = r'''Event-level performance on the synthetic test sequence containing 2,000 ground-truth events.
Events are matched one-to-one within the common in-plane and temporal gate defined in Sec.~\ref{sec:synthetic_hologram_generation}.
AP is the integral of the monotonically enveloped precision--recall curve, with tied scores included together, and the maximum \(F_1\) score is obtained by varying the score threshold.
Precision at each target recall is evaluated at the highest score threshold attaining or exceeding that recall.
Candidate-pool recall is evaluated for the generated pool \(\mathcal{H}\), before candidate-wise scoring and duplicate suppression.
Values in parentheses denote the difference from Ours (method minus Ours), calculated before rounding.'''

TABLE5_CAPTION = r'''Comparison of scoring stages of the proposed method on the synthetic test sequence containing 2,000 ground-truth events.
All rows include temporal-mode aggregation, a cap of five modes per terminating trajectory, rank adjustment using ranks fixed before suppression, and the same trained localization regressors.
The trajectory-only row uses the static shortlist without image features, whereas the subsequent rows use the scoring union and the image features available at those stages.
Coordinate-based duplicate suppression is added only in the final row.
The comparison therefore includes changes in candidate selection and localization-feature availability, as well as scoring.
At each target recall, precision and false-positive counts are evaluated at the highest score threshold attaining or exceeding that recall, including tied scores jointly.
AP denotes average precision, P precision, R recall, and FP the number of false positives.'''


def method_edits():
    """Exact source anchors; keep these distinct from numerical result replacement."""
    return [
        {
            'id': 'reconstruction-range', 'title': '生成奥行きと再構成距離',
            'start': 'The distance to the front reconstruction plane was',
            'end': 'The recording wavelength, pixel pitch, image dimensions, and reconstruction settings were the same as those used in the experiment (Table~\\ref{tab:exp_param}).',
            'replacement': r'''The generated volume extended from \(z=0\) to \ensuremath{102.3\,\unit{mm}} relative to its front plane.
Particle reconstruction sampled 1,024 axial planes at \ensuremath{0.1\,\unit{mm}} intervals, corresponding to propagation distances of \ensuremath{74.9\,\unit{mm}} to \ensuremath{177.2\,\unit{mm}} from the primary hologram plane.
The hologram-plane separation was \ensuremath{21.45\,\unit{mm}}.
Three phase-retrieval iterations were used throughout particle measurement and image-feature extraction.
The recording wavelength, pixel pitch, and image dimensions were the same as those used in the experiment (Table~\ref{tab:exp_param}).''',
            'note': '0–102.3 mmは体積内座標。カメラからの距離は74.9–177.2 mm。パディングの本文追記は不要。',
        },
        {
            'id': 'test-provenance', 'title': 'テスト利用履歴とFig.7の位置づけ',
            'start': 'Five synthetic sequences were used for training,',
            'end': 'Figure~\\ref{fig:synthetic_hologram_distribution} shows the particle detection and tracking performance of HoloD3 \\citep{HoloD3} on this test sequence.',
            'replacement': r'''Five synthetic sequences (seeds 260740--260744) were used for training, and the remaining sequence (seed 260745) was used for evaluation without model fitting.
The same test seed had previously been used for implementation and sensitivity checks on the earlier generated data.
The settings for the present full evaluation were fixed before inspecting its results; this sequence should therefore not be interpreted as a previously untouched test set.
Figure~\ref{fig:synthetic_hologram_distribution} shows the particle detection and tracking performance on this sequence, using the same reconstruction and tracking outputs supplied to the collision-detection evaluation in Fig.~\ref{fig:synthetic_hologram_validation_results}.
The particle precision--recall curve additionally uses detections at lower confidence thresholds from the same reconstruction setup.''',
            'note': '学習への不使用と、過去に一度も参照していないことを区別する。低confidenceのPR評価と追跡入力0.1も区別。',
        },
        {
            'id': 'training-labels', 'title': 'R5：親・娘ID、neutral、学習Zゲート',
            'start': 'Training candidates were first matched to ground-truth events using particle IDs.',
            'end': 'The weights were further normalized so that the total weights of positive and negative examples were equal.',
            'replacement': r'''Within the candidate set used by each training stage, candidates were first matched to ground-truth events using particle identities.
For a candidate at frame \(t\), an identity-consistent match required observations of both parents at \(t-1\) and the daughter at \(t+1\), without imputing missing observations.
Spatial eligibility required an in-plane distance of at most \ensuremath{350\,\unit{\mu m}} and a time difference of at most five frames; for Ours, an axial distance of at most \ensuremath{15\,\unit{mm}} was additionally required.
The depth-free variants did not use this axial gate.
If a ground-truth event had an identity-consistent candidate in the current training set, spatially eligible but identity-inconsistent candidates for that event were assigned zero loss weight.
For events without an identity-consistent candidate, spatially eligible candidates were used as positive examples.
Positive weights were normalized to give each represented ground-truth event equal total weight across the training sequences, and the positive and negative class totals were balanced.
The identity and fallback assignments and event and class weights were recomputed for the retained top-20 inputs to the relation-attention model.''',
            'note': '本文で学習の±15 mmと評価のZ条件なしを混同しない。規則は各学習段階の候補集合に適用。',
        },
        {
            'id': 'rank-order', 'title': 'R4：抑制前の順位を保持',
            'start': 'For each terminating trajectory, the temporal-mode representatives remaining after duplicate suppression are ranked in descending order of $u_i$.',
            'end': 'For each terminating trajectory, the temporal-mode representatives remaining after duplicate suppression are ranked in descending order of $u_i$.',
            'replacement': r'''For each terminating trajectory, temporal-mode representatives are ranked in descending order of \(u_i\) before applying the mode cap and coordinate-based duplicate suppression.
These ranks are retained for the final score adjustment; surviving representatives are not reranked.''',
            'note': '順位式自体は維持。感度分析では手法により優劣が異なるため、普遍的な優位性を主張しない。',
        },
        {
            'id': 'ap-definition', 'title': 'APの定義と同点処理',
            'start': 'The reported metrics therefore quantify event detection performance within the specified in-plane and temporal matching tolerances.',
            'end': 'The reported metrics therefore quantify event detection performance within the specified in-plane and temporal matching tolerances.',
            'replacement': r'''The reported metrics quantify event detection performance within the specified in-plane and temporal matching tolerances.
Predictions with tied scores enter together at each threshold.
Average precision (AP) is computed as the sum of recall increments weighted by the monotonically enveloped precision values, where the precision envelope at a given recall is the maximum precision attained at that or any higher recall.
This all-points interpolated definition is distinct from the non-interpolated average-precision calculation in scikit-learn.''',
            'note': '計算定義は公開版から維持。結果本文・表キャプションのscikit-learn APとしての引用も差し替え案で修正。',
        },
        {
            'id': 'generator-provenance', 'title': '合成生成に用いた実験統計の固定',
            'start': 'The particle statistics used to generate the synthetic fields were based on trajectory statistics from the 14 experimental sequences described in Sec.~\\ref{sec:exp_setup}.',
            'end': 'The particle statistics used to generate the synthetic fields were based on trajectory statistics from the 14 experimental sequences described in Sec.~\\ref{sec:exp_setup}.',
            'replacement': r'''The empirical diameter and velocity distributions used to generate the synthetic fields were frozen from the documented earlier processing of the 14 experimental sequences described in Sec.~\ref{sec:exp_setup}.
Subsequent reprocessing of the experimental measurements does not retroactively change these generator inputs.''',
            'note': '実験データ再処理後のFig.3–5と、今回の合成の入力になった分布の版を区別。鉛直速度摂動は維持。',
        },
        {
            'id': 'results-introduction', 'title': 'Results導入部の段階比較の呼び方',
            'start': 'Comparisons with MinIP-based 2D and the depth-free variant assess, respectively,',
            'end': 'A cumulative ablation study further evaluates the contributions of successive processing stages within the proposed method.',
            'replacement': r'''Comparisons with MinIP-based 2D and the depth-free variant assess, respectively, the benefit of the proposed method over projection-only analysis and the additional contribution of depth information during collision detection.
A comparison of scoring stages further evaluates the combined effects of candidate selection, available features, scoring, and coordinate-based duplicate suppression within the proposed method.''',
            'note': '候補集合と位置補正入力も変わるため、スコアだけの独立した寄与の実験とは書かない。',
        },
    ]


def result_paragraphs(n):
    m = n['methods']; o, d, p = (m[k] for k in ('ours', 'depth_free', 'minip2d'))
    f = lambda v: f'{v:.3f}'
    fp_reductions = n['false_positive_reductions']
    beginning = rf'''\subsection{{Method validation with synthetic holograms}}
Under the common spatiotemporal matching criterion, the proposed method achieved higher AP, maximum \(F_1\), and precision at the specified recall levels than both baselines (Table~\ref{{tab:projected_xy_metric_summary}}).
AP was {f(o['event_ap'])} for Ours, {f(d['event_ap'])} for the depth-free variant, and {f(p['event_ap'])} for MinIP-based 2D.
The corresponding maximum \(F_1\) scores were {f(o['maximum_f1'])}, {f(d['maximum_f1'])}, and {f(p['maximum_f1'])}.
Precision at a recall of 0.8 was {f(o['precision_at_recall_0p8'])}, {f(d['precision_at_recall_0p8'])}, and {f(p['precision_at_recall_0p8'])}, respectively; at a recall of 0.9, the corresponding values were {f(o['precision_at_recall_0p9'])}, {f(d['precision_at_recall_0p9'])}, and {f(p['precision_at_recall_0p9'])}.
Candidate-pool recall was {f(o['candidate_pool_recall'])}, {f(d['candidate_pool_recall'])}, and {f(p['candidate_pool_recall'])}, respectively.
The maximum recall after final processing was {f(o['maximum_recall'])} for Ours and {f(d['maximum_recall'])} for both baselines.
Thus, the precision advantage of Ours does not imply an advantage in candidate-pool or maximum final recall.
AP and maximum \(F_1\) summarize detection performance over score thresholds, whereas precision and false-positive counts at target recalls directly describe the candidate review burden.'''
    comparison = rf'''Figure~\ref{{fig:synthetic_hologram_validation_results}} shows detection performance and review burden for the same test sequence containing 2,000 ground-truth events.
Candidates were classified as true or false positives using the event-matching rule, without visual inspection.
At a recall of 0.8, the false-positive counts were {o['false_positives_at_recall_0p8']} for Ours, {d['false_positives_at_recall_0p8']} for the depth-free variant, and {p['false_positives_at_recall_0p8']} for MinIP-based 2D.
These correspond to 1,651, 1,693, and 1,889 reviewed candidates, respectively, including 1,600 matched events.
Relative to MinIP-based 2D, the depth-free variant reduced false positives by {fp_reductions['minip2d_to_depth_free_false_positives_at_recall_0p8_percent']:.1f}\%, and Ours reduced them by {fp_reductions['minip2d_to_ours_false_positives_at_recall_0p8_percent']:.1f}\%.
Ours reduced false positives by a further {fp_reductions['depth_free_to_ours_false_positives_at_recall_0p8_percent']:.1f}\% relative to the depth-free variant.
At a recall of 0.9, the counts were {o['false_positives_at_recall_0p9']}, {d['false_positives_at_recall_0p9']}, and {p['false_positives_at_recall_0p9']:,}, respectively.

At recall 0.8, the numbers of candidates with no ground-truth event within the matching gate were 43, 87, and 268 for Ours, the depth-free variant, and MinIP-based 2D; the corresponding duplicate or competing counts were 8, 6, and 21.
Most of the net reduction in each comparison with Ours therefore arose from candidates without a nearby ground-truth event.
This result is consistent with depth information helping to lower the ranks of trajectory pairs that are close in projection but separated axially.

MinIP-based 2D and the depth-free variant both use MinIP images for image-feature extraction and exclude depth during collision detection.
Their comparison assesses the combined effect of replacing MinIP-derived diameter estimates and independently linked two-dimensional trajectories with in-focus diameter estimates and depth-assisted trajectories.
The depth-free variant's intermediate precision supports the benefit of these upstream observations, and the additional improvement with Ours supports using depth during collision detection on this sequence.'''
    a, b, c, z = n['table4']
    stage = rf'''Table~\ref{{tab:projected_xy_stage_ablation}} compares scoring stages under common temporal-mode aggregation, a per-trajectory mode cap, rank adjustment, and localization correction.
Candidate selection and the features available to the localization regressors also differ between the trajectory-only and image-feature rows, so this comparison does not isolate changes to classifier scores alone.
AP increased from {f(a['event_ap'])} to {f(b['event_ap'])}, {f(c['event_ap'])}, and {f(z['event_ap'])} across the four rows.
From the first to the final row, false positives decreased from {a['false_positives_at_recall_0p8']} to {z['false_positives_at_recall_0p8']} at recall 0.8 ({100*(1-z['false_positives_at_recall_0p8']/a['false_positives_at_recall_0p8']):.1f}\%) and from {a['false_positives_at_recall_0p9']:,} to {z['false_positives_at_recall_0p9']} at recall 0.9 ({100*(1-z['false_positives_at_recall_0p9']/a['false_positives_at_recall_0p9']):.1f}\%).

The image-feature stage increased AP and reduced false positives at recall 0.9 from {a['false_positives_at_recall_0p9']:,} to {b['false_positives_at_recall_0p9']:,}.
However, false positives at recall 0.8 increased from {a['false_positives_at_recall_0p8']} to {b['false_positives_at_recall_0p8']}, and maximum \(F_1\) decreased slightly from {a['maximum_f1']:.4f} to {b['maximum_f1']:.4f}.
Improvements were therefore not uniform across operating points.
Relation-attention subsequently reduced false positives to {c['false_positives_at_recall_0p8']} and {c['false_positives_at_recall_0p9']} at recalls 0.8 and 0.9; coordinate-based duplicate suppression reduced them further to {z['false_positives_at_recall_0p8']} and {z['false_positives_at_recall_0p9']}.
The last-row comparison concerns coordinate-based suppression; temporal-mode aggregation, the mode cap, and rank adjustment are already present in every row.'''
    true, artifact = n['figure9_cases']
    examples = rf'''Figure~\ref{{fig:image_feature_case_study}} illustrates two individual scoring outcomes.
For the true collision--coalescence event, the trajectory and image-feature scores were {true['trajectory_score']:.6f} and {true['image_feature_score']:.6f}, respectively.
For the shared-particle artifact, the score decreased from {artifact['trajectory_score']:.6f} to {artifact['image_feature_score']:.6f}.
The two reconstructed trajectory IDs were supported by the same ground-truth particle at different preceding frames, and their image features indicated a shared connected component in all nine evaluated pre-event frames.
The true event was retained and the artifact was excluded by the final candidate-selection policy.
The examples were selected after examining the results for visible particle separation before a true event and overlapping track observations in a shared-particle artifact; they do not estimate an average image-feature effect.'''
    ending = r'''The comparisons support the benefits of depth-informed particle measurements and trajectories, and of depth use during collision detection, for precision on this synthetic sequence.
The scoring-stage comparison supports the final combination while also showing that individual metrics need not improve at every stage.
Because the synthetic fields omit some experimental noise, particle-density variation, and collision-induced deformation, these numerical results do not directly estimate experimental performance or missed-event rates.
The prior use of the test seed for implementation and sensitivity checks is an additional limitation; an evaluation on previously unused sequences would be needed to assess generalization without that prior use.
Experimental evaluation is therefore reported separately.'''
    return beginning, comparison, stage, examples, ending
