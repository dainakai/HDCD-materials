"""Trace every Fig.13 exclusion to observed source-track frames, without imputation."""
import html
import numpy as np
import pandas as pd
from common import *

ROLES=[('pre_vanished','vanished_track_id','pre','pre_diameter_vanished','pre_vy_vanished','A（消失軌跡）の前窓'),
       ('pre_survivor','survivor_track_id','pre','pre_diameter_survivor','pre_vy_survivor','B（存続軌跡）の前窓'),
       ('post_survivor','survivor_track_id','post','post_diameter','post_vy','B（存続軌跡）の後窓')]


def frames_text(values):
    values=[int(x) for x in values]
    if not values:return 'なし'
    groups=[];start=end=values[0]
    for value in values[1:]:
        if value==end+1:end=value
        else:groups.append(str(start) if start==end else f'{start}–{end}');start=end=value
    groups.append(str(start) if start==end else f'{start}–{end}')
    return ', '.join(groups)


def reason(w):
    role=w['role_ja'];actual=w['velocity_observed_points'];f=w['candidate_frame_raw']
    if w['diameter_valid_points']==0:
        if w['diameter_selected_points']==0:
            if w['when']=='pre':
                return f'{role}：軌跡の開始はf={w["track_start_frame"]}で、前窓の上限f−2={f-2}以前に観測がない。'
            return f'{role}：軌跡の終了はf={w["track_end_frame"]}で、候補f={f}より後の観測がない。'
        return f'{role}：選択した{w["diameter_selected_points"]}点の粒径がすべて非有限または非正。'
    if actual<3:
        if w['when']=='pre' and w['eligible_observations']<3:
            return f'{role}：軌跡はf={w["track_start_frame"]}から始まり、f−2={f-2}以前の観測は{w["eligible_observations"]}点のみ。速度窓の実観測は{actual}点（f={w["velocity_frames_text"]}）で必要な3点に足りない。'
        if w['when']=='post' and w['eligible_observations']<3:
            return f'{role}：候補後の観測は{w["eligible_observations"]}点で軌跡がf={w["track_end_frame"]}に終わる。速度窓の実観測は{actual}点（f={w["velocity_frames_text"]}）で必要な3点に足りない。'
        return f'{role}：粒径から定めたf={w["velocity_start_frame"]}–{w["velocity_end_frame"]}の速度窓に、有効な鉛直位置が{actual}点（f={w["velocity_frames_text"]}）しかない。'
    return ''


def main():
    events=pd.read_parquet(PRIVATE/'physical_events_all.parquet')
    freeze=read(PRIVATE/'freeze.json')
    expected={r['path']:r['sha256'] for r in freeze['sources']}
    records=[];windows=[];sources=[]
    for scene in SCENES:
        path=CAMPAIGN/f'scenes/scene{scene:04d}/tracking/smoothed.parquet'
        digest=sha(path);assert digest==expected[str(path.relative_to(WORK))]
        groups={int(k):v.sort_values('frame') for k,v in pd.read_parquet(path).groupby('track_id',sort=False)}
        raw_path=path.with_name('raw.parquet')
        raw=pd.read_parquet(raw_path,columns=['track_id','frame'])
        raw_groups={int(k):v.sort_values('frame') for k,v in raw.groupby('track_id',sort=False)}
        sources.append(dict(scene=scene,smoothed_sha256=digest,raw_sha256=sha(raw_path)))
        for e in events.loc[events.scene_num.eq(scene)].itertuples():
            event_windows=[];f=int(e.candidate_frame_raw)
            for role,tidcol,when,dcol,vcol,role_ja in ROLES:
                tid=int(getattr(e,tidcol));track=groups[tid];raw_track=raw_groups[tid]
                assert track.frame.tolist()==raw_track.frame.tolist(),(e.candidate_id,role,'smoothing changed observed frame support')
                eligible=track.loc[track.frame.le(f-2) if when=='pre' else track.frame.gt(f)]
                selected=eligible.tail(10) if when=='pre' else eligible.head(20)
                good=selected.loc[np.isfinite(selected.diameter_px)&selected.diameter_px.gt(0)]
                observed_d=float(good.diameter_px.median()*10) if len(good) else np.nan
                np.testing.assert_allclose(observed_d,getattr(e,dcol+'_um'),rtol=1e-12,atol=1e-10,equal_nan=True)
                assert len(good)==getattr(e,dcol+'_n')
                start,end=getattr(e,when+'_window_start_frame'),getattr(e,when+'_window_end_frame')
                segment=track.loc[track.frame.between(start,end)] if np.isfinite([start,end]).all() else track.iloc[:0]
                valid=segment.loc[np.isfinite(segment.y)].drop_duplicates('frame',keep='last')
                vf=valid.frame.to_numpy(dtype=int)
                api_n=len(valid) if len(valid)>=2 else 0
                assert api_n==getattr(e,vcol+'_n'),(e.candidate_id,role)
                result=dict(candidate_id=e.candidate_id,scene_num=scene,review_rank=int(e.review_rank),status=e.status,
                    candidate_frame_raw=f,role=role,role_ja=role_ja,track_id=tid,when=when,
                    track_start_frame=int(track.frame.min()),track_end_frame=int(track.frame.max()),track_points=len(track),
                    raw_and_smoothed_frames_identical=True,eligible_observations=len(eligible),
                    diameter_selected_points=len(selected),diameter_valid_points=len(good),
                    diameter_frames_text=frames_text(good.frame),diameter_invalid_points=len(selected)-len(good),
                    velocity_start_frame=int(start) if np.isfinite(start) else None,velocity_end_frame=int(end) if np.isfinite(end) else None,
                    velocity_observed_points=len(valid),velocity_api_points=api_n,
                    velocity_invalid_y_points=int((~np.isfinite(segment.y)).sum()),velocity_intervals=max(len(valid)-1,0),
                    velocity_frames_text=frames_text(vf),velocity_frame_list=';'.join(str(x) for x in vf),
                    velocity_max_frame_gap=int(np.diff(vf).max()) if len(vf)>1 else None,
                    diameter_ok=bool(len(good)),velocity_window_ok=bool(api_n>=3))
                result['reason_ja']=reason(result)
                event_windows.append(result);windows.append(result)
            ds=all(x['diameter_ok'] for x in event_windows)
            vs=all(x['velocity_window_ok'] for x in event_windows) and all(x['diameter_ok'] for x in event_windows[:2])
            assert ds==e.diameter_available and vs==e.velocity_available,e.candidate_id
            binary=e.status!='ambiguous'
            dflags=[x['role']+'_diameter_missing' for x in event_windows if not x['diameter_ok']]
            vflags=[x['role']+'_fewer_than_3_points' for x in event_windows if not x['velocity_window_ok']]
            reasons=[x['reason_ja'] for x in event_windows if x['reason_ja']]
            record=dict(candidate_id=e.candidate_id,scene_num=scene,scene_index=SCENES.index(scene)+1,review_rank=int(e.review_rank),
                candidate_frame_raw=f,status=e.status,diameter_available=ds,velocity_available=vs,
                fig13a_included=binary and ds,fig13b_included=binary and vs,
                diameter_exclusion_flags=';'.join(([] if binary else ['ambiguous_label'])+dflags),
                velocity_exclusion_flags=';'.join(([] if binary else ['ambiguous_label'])+[flag for flag in dflags if flag.startswith('pre_')]+vflags),
                measurement_reason_ja=' '.join(reasons) or '粒径・速度の測定条件を満たす。',
                label_reason_ja='' if binary else '人の判定がambiguousのため、二値判定のFig.13(a)(b)から除外。',
                actual_velocity_points='/'.join(str(x['velocity_observed_points']) for x in event_windows),
                recorded_velocity_api_points='/'.join(str(x['velocity_api_points']) for x in event_windows),
                diameter_points='/'.join(str(x['diameter_valid_points']) for x in event_windows))
            records.append(record)
        print(f'Audited source windows: S{scene:04d}',flush=True)
    audit=pd.DataFrame(records);ws=pd.DataFrame(windows)
    audit.to_csv(DATA/'figure13_eligibility_all1120.csv',index=False)
    excluded=audit.loc[~audit.fig13a_included|~audit.fig13b_included]
    excluded.to_csv(DATA/'figure13_excluded_samples.csv',index=False)
    ws.to_csv(DATA/'figure13_window_evidence.csv',index=False)
    counts=[]
    for status,group in audit.groupby('status',sort=False):
        for panel,col in [('a','fig13a_included'),('b','fig13b_included')]:
            counts.append(dict(status=status,panel=panel,total=len(group),included=int(group[col].sum()),excluded=int((~group[col]).sum())))
    pd.DataFrame(counts).to_csv(DATA/'figure13_exclusion_counts.csv',index=False)
    assert audit.fig13a_included.sum()==1092 and audit.fig13b_included.sum()==1001
    assert (excluded.status=='collision').sum()==7
    summary=dict(created_at=stamp(),reviewed=1120,counts=counts,unique_excluded_candidates=len(excluded),
        invalid_diameter_points=int(ws.diameter_invalid_points.sum()),invalid_y_points=int(ws.velocity_invalid_y_points.sum()),
        one_observation_windows_reported_as_zero_by_api=int((ws.velocity_observed_points.eq(1)&ws.velocity_api_points.eq(0)).sum()),
        raw_and_smoothed_observation_support_identical=True,
        technical_reason_counts={status:{role:int((group.role.eq(role)&~group.velocity_window_ok).sum()) for role,*_ in ROLES}
                                 for status,group in ws.groupby('status')},
        sources=sources,analysis_code_sha256=sha(Path(__file__)),
        interpretation='Reasons are observed track-support limitations under the fixed raw-time windows, not a new human judgment or an inferred physical cause of track onset/termination.')
    save(DATA/'figure13_exclusion_summary.json',summary)
    write_report(audit,ws,summary)
    print(summary['counts'])


def write_report(audit,ws,summary):
    from build_report import table
    exclusions=audit.loc[~audit.fig13a_included|~audit.fig13b_included]
    counts=table(['判定','全件','粒径(a) 採用 / 除外','速度(b) 採用 / 除外'],[
        [name,len(g),f'{g.fig13a_included.sum()} / {(~g.fig13a_included).sum()}',f'{g.fig13b_included.sum()} / {(~g.fig13b_included).sum()}']
        for status,name in [('collision','確認衝突'),('non_collision','非衝突'),('ambiguous','曖昧')]
        for g in [audit.loc[audit.status.eq(status)]]])
    tp=exclusions.loc[exclusions.status.eq('collision')]
    short=table(['原系列 / 順位','候補 f','粒径の点数 A前/B前/B後','速度の実観測点数','除外されるパネル'],[
        [f'S{r.scene_num:04d} / {r.review_rank}位',r.candidate_frame_raw,r.diameter_points,r.actual_velocity_points,
         '粒径・速度' if not r.fig13a_included else '速度'] for r in tp.itertuples()])
    def detail(r):
        rows=ws.loc[ws.candidate_id.eq(r.candidate_id)]
        evidence=table(['窓','track ID','軌跡の全範囲','粒径の有効 f','速度の有効 f','実点数 / API点数'],[
            [w.role_ja,w.track_id,f'{w.track_start_frame}–{w.track_end_frame} ({w.track_points}点)',w.diameter_frames_text,w.velocity_frames_text,
             f'{w.velocity_observed_points} / {w.velocity_api_points}'] for w in rows.itertuples()])
        reason_text='\n'.join(f'<p>{html.escape(t)}</p>' for t in [r.label_reason_ja,r.measurement_reason_ja] if t)
        return f'<details id="case-{r.scene_num}-{r.review_rank}"><summary>S{r.scene_num:04d} / {r.review_rank}位 · f={r.candidate_frame_raw} · 粒径 {"採用" if r.fig13a_included else "除外"} / 速度 {"採用" if r.fig13b_included else "除外"}</summary>{reason_text}<p class="muted">候補ID: <code>{r.candidate_id}</code></p>{evidence}</details>'
    groups=''
    for status,name in [('collision','確認衝突7件'),('non_collision','非衝突の全除外例'),('ambiguous','曖昧判定12件')]:
        subset=exclusions.loc[exclusions.status.eq(status)]
        groups+=f'<section id="{status}"><h2>{name}</h2>'+''.join(detail(r) for r in subset.itertuples())+'</section>'
    reason_rows=[]
    for status,name in [('collision','確認衝突'),('non_collision','非衝突'),('ambiguous','曖昧')]:
        for role,*rest in ROLES:
            g=ws.loc[ws.status.eq(status)&ws.role.eq(role)]
            reason_rows.append([name,rest[-1],int((~g.diameter_ok).sum()),int((~g.velocity_window_ok).sum())])
    reason_table=table(['判定','対象窓','粒径なし','速度3点未満'],reason_rows)
    content=f'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fig.13 集計対象外の全サンプル</title><link rel="stylesheet" href="report.css"></head><body>
<header><div class="eyebrow">FIGURE 13 · SOURCE-TRACK AUDIT</div><h1>各サンプルが集計に入らない理由</h1><p>1,120件すべてについて、保存された判定と元の軌跡の観測フレームを照合しました。
粒径図から28件、速度図から119件が除外されます。どちらかから除外される候補の合計は{len(exclusions)}件です。</p>
<div class="buttons"><a class="button" href="index.html#fig13">図表報告に戻る</a><a class="button" href="data/figure13_excluded_samples.csv">全除外例 CSV</a><a class="button" href="data/figure13_window_evidence.csv">各窓の根拠 CSV</a></div></header>
<nav><div><a href="#rules">判定条件</a><a href="#collision">確認衝突7件</a><a href="#non_collision">非衝突</a><a href="#ambiguous">曖昧</a></div></nav><main>
<section id="rules"><h2>除外は判定ラベルと測定窓の条件で決まる</h2>{counts}
<p>粒径図(a)には、親2粒子と衝突後の粒径がすべて有限かつ正である候補を使います。
速度図(b)には、親2粒径が得られ、Aの前・Bの前・Bの後の3窓でそれぞれ3点以上の観測から速度を計算できる候補を使います。
曖昧判定12件は、測定値の有無にかかわらず両図の対象外です。</p>
<p>A・Bは候補の軌跡仮説における役割です。「存続側B」という名称だけで候補時刻後の観測が保証されるわけではないため、実際に後窓の観測があるかを確認します。</p>
<p>候補時刻をfとすると、親粒径はf−2以前の最後の10観測、衝突後粒径はfより後の最初の20観測から求めます。
速度の前窓は、2つの親粒径窓の開始・終了を合わせた範囲です。そのため、片方の軌跡で10点を超える場合もあります。
観測フレームの欠落は補間せず、隣接する観測の変位を実際のフレーム間隔で割ります。</p>
<div class="notice"><strong>保存された速度の「点数0」は、観測が0点とは限りません。</strong><p>速度APIは1点だけの場合も「速度なし・点数0」を返します。
今回の表は元の軌跡を読み、実際の0点・1点・2点を区別しました。
全3,360窓のうち{summary['one_observation_windows_reported_as_zero_by_api']}窓にこの表現の違いがあります。</p></div>
<p>選択された粒径値の非有限・非正は{summary['invalid_diameter_points']}点、速度窓の非有限な鉛直位置は{summary['invalid_y_points']}点でした。
raw軌跡と平滑化後の軌跡の観測フレームは全対象で一致し、平滑化によるフレーム削除はありません。
測定条件を満たさない直接の理由は、所定の時間範囲に軌跡の観測が足りないことです。</p>
<p>軌跡がそこで始まる・終わることは確認できますが、その物理的な原因までは表だけで断定できません。
画角への出入り、粒子の見失い、軌跡の分断などの切り分けには、計測前の画像と検出結果の追加調査が必要です。
この調査では時間窓・軌跡・人の判定を変更していません。</p></section>
<section><h2>確認衝突186件のうち集計から外れる7件</h2>{short}<p>確認衝突7件すべてで速度窓が不足し、そのうち2件は粒径窓も欠けています。
各候補の詳しい観測フレームは下の項目を開いて確認できます。</p>{reason_table}<p class="muted">理由の窓別集計は重複を含みます。同じ候補で複数の窓が不足するため、列の和は除外候補数になりません。</p></section>
{groups}<section><h2>再確認用のファイル</h2><ul><li><a href="data/figure13_eligibility_all1120.csv">1,120件の採用・除外判定</a></li><li><a href="data/figure13_exclusion_summary.json">集計と元軌跡のSHA-256</a></li><li><a href="data/figure13_exclusion_counts.csv">パネル・判定別の件数</a></li></ul></section>
<footer>生成: {stamp()} · 元の判定・物理診断の集計値は変更していません。</footer></main></body></html>'''
    (PACKAGE/'fig13-exclusions.html').write_text(content)


if __name__=='__main__':main()
