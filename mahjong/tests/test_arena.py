import pytest
from mahjong.arena import cluster_stats,summarize


def test_wall_is_statistical_unit_and_paired_difference():
    rows=[]
    for candidate,delta in [('A',0),('B',2)]:
        for seed in range(20):
            for seat in range(4):
                rows.append(dict(candidate=candidate,opponent='random',seed=seed,seat=seat,wall_id=str(seed),
                                 reward=seed+delta,score_difference=seed+delta,win=1,loss=0,draw=0))
    r=summarize(rows)
    assert r['summary']['A vs random']['reward']['walls']==20
    assert r['paired_differences']['B - A vs random']['ci95']==[2.,2.]
    assert cluster_stats(range(20))==cluster_stats(range(20))
    with pytest.raises(ValueError):summarize(rows[:-1])
