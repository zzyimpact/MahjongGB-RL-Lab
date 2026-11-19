import json
import pytest
from mahjong.analysis_v3 import followup


def test_followup_adoption_direction_and_population_identity(tmp_path):
    def write(name,data):
        p=tmp_path/name;p.write_text(json.dumps(data));return p
    plan=dict(seed=123,walls=4,candidates={'PPO':{'sha256':'p'},'BC':{'sha256':'b'}},primary_contrast='PPO - BC',bootstrap_seed=7,bootstrap_replicates=200)
    plan_path=write('plan.json',plan)
    cfg=dict(purpose='confirmation',seed_start=123,walls=4,candidates=['PPO','BC'],opponents=['random','teacher'],identities=plan['candidates'])
    write('arena_config.json',cfg)
    def results(delta):
        return write('results.json',dict(wall_records=[dict(candidate=n,opponent=o,seed=s,reward=delta if n=='PPO' else 0) for n in ['PPO','BC'] for o in cfg['opponents'] for s in range(123,127)]))
    output=tmp_path/'analysis.json'
    assert followup(results(1),plan_path,output)['recommended_model']=='PPO'
    assert followup(results(-1),plan_path,output)['recommended_model']=='BC'
    cfg['seed_start']=124;write('arena_config.json',cfg)
    with pytest.raises(ValueError,match='population'):followup(results(1),plan_path,output)
