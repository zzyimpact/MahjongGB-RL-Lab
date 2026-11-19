"""Conservative PPO using the verified collector/GAE/learner.

Two learning seats rotate against two deterministic, frozen opponents. The
critic never backpropagates into the actor backbone; initial critic-only
updates leave every actor tensor bit-identical to BC.
"""
import argparse
import copy
from functools import partial
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from rlframework.collector import PolicyRef
from rlframework.collectors.routing import StaticPolicyRouter
from rlframework.collectors.parallel import ParallelEpisodeCollector
from rlframework.collectors.sparse import SparseDecisionCollector
from rlframework.collectors.rewards import SharedTensorRewardResolver
from rlframework.algorithm.named_policy import NamedCategoricalPolicy
from rlframework.algorithm.ppo import PPOLearner,PPOConfig
from rlframework.processing.gae import GAEProcessor
from rlframework.workflow.on_policy import MultiPolicyOnPolicyWorkflow,OnPolicyStack
from rlframework.data import Collection
from .arena import load_checkpoint,identity
from .arena_v3 import run as run_arena, wall_hash, HISTORICAL_RANGES
from .policy import make_policy
from .env import Environment
from .model import input_keys,model_spec
from .ppo_anchor import BCReferenceLoss
from .train_v3 import atomic_save


def isolate_critic(model):
    # Applies to tiny, legacy ResNet and V3 without changing checkpoint keys.
    return model.critic.register_forward_pre_hook(lambda module,args:(args[0].detach(),))


class FrozenOpponent(NamedCategoricalPolicy):
    def __init__(self,path,available_features):
        model,cfg=load_checkpoint(path);super().__init__(model,input_keys=input_keys(cfg))
        self.basic=cfg.get('feature_set','basic')=='basic'
        if cfg.get('feature_set')=='v3' and available_features!='v3':raise ValueError('V3 opponent needs V3 observation')
        if not self.basic and available_features=='basic':raise ValueError('Enhanced opponent needs enhanced observation')
        self.eval()
        for p in self.parameters():p.requires_grad_(False)
    def _forward(self,observation):
        # This is an explicit, tested historical feature adapter: v1 is exactly
        # the first six v2/v3 planes. Models themselves remain strict-loading.
        if self.basic:
            observation=observation.clone();x=observation.get('obs')
            observation.set('obs',x[..., :6,:,:])
        return super()._forward(observation)
    def act(self,observation,state=None,deterministic=False):
        return super().act(observation,state,deterministic=True)


class MixedCollector:
    def __init__(self,worker_id,config,opponents,mode='mixed'):
        self.policy=make_policy(config=config);self.environment=Environment(feature_set=config.get('feature_set','basic'))
        self.router=StaticPolicyRouter({str(i):PolicyRef('shared',self.policy) for i in range(4)})
        self.frozen={k:FrozenOpponent(p,config.get('feature_set','basic')) for k,p in opponents.items()}
        self.episode_action_seeds=config.get('rl',{}).get('episode_action_seeds',False)
        self.mode=mode;self.gamma=config['ppo']['gamma'];self.gae_lambda=config['ppo']['gae_lambda']
        self.inner=SparseDecisionCollector(self.environment,self.router,SharedTensorRewardResolver('rewards',{str(i):i for i in range(4)}),
            gamma=self.gamma,gae_lambda=self.gae_lambda,env_id='actor_%d'%worker_id)
        self.episode_stats=[]
    def collect_episode(self,config=None,deterministic=False):
        seed=int(config['seed'])
        if getattr(self,'episode_action_seeds',False):torch.manual_seed(int(np.random.SeedSequence([seed,577]).generate_state(1)[0]))
        names=sorted(self.frozen);name=names[(seed//4)%len(names)] if names else 'self'
        learning={seed%4,(seed+1)%4} if self.mode=='mixed' and names else set(range(4))
        version=self.router.versions['shared']
        self.inner.router=StaticPolicyRouter({str(i):PolicyRef('shared',self.policy,version) if i in learning else PolicyRef('frozen_'+name,self.frozen[name],0) for i in range(4)})
        batch=self.inner.collect_episode(config=config,deterministic=deterministic)
        rows=[t for t in batch.trajectories if t.policy_id=='shared']
        self.episode_stats.append(dict(seed=seed,opponent=name if self.mode=='mixed' else 'self',learning_seats=sorted(learning),
            scores=list(self.environment.core.reward or [0]*4),wall_id=hashlib.sha256(self.environment.core.originalTileWall.encode()).hexdigest(),
            all_decisions=batch.agent_steps,learning_decisions=sum(len(t.data) for t in rows)))
        return Collection(rows,batch.env_steps,batch.agent_steps,batch.completed_episodes)


def factory(worker_id,config,opponents,mode):
    torch.manual_seed(config['seed']+worker_id)
    return MixedCollector(worker_id,config,opponents,mode)


def train(config_file,checkpoint,output,resume=False):
    cfg=json.loads(Path(config_file).read_text());base=torch.load(checkpoint,map_location='cpu');out=Path(output);rl=cfg['rl']
    if model_spec(cfg)!=model_spec(base['config']):raise ValueError('BC model identity mismatch')
    if out.exists() and not resume:raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    policy=make_policy(config=cfg);policy.load_state_dict(base['policy'],strict=True)
    isolate_critic(policy.module)
    learner=PPOLearner(policy,PPOConfig(**cfg['ppo']),device=cfg['device'])
    learner.auxiliary_loss=BCReferenceLoss(policy,checkpoint,cfg,cfg['device'],rl['bc_kl'])
    frozen=dict(rl['opponents']);frozen['BC']=str(checkpoint)
    training_walls=[wall_hash(s) for s in range(cfg['seed'],cfg['seed']+cfg['iterations']*cfg['actors']*cfg['episodes_per_actor'])]
    excluded=set(json.loads(Path('reports/round3/raw/split_manifest.json').read_text())['groups'])
    for start,n in HISTORICAL_RANGES:excluded.update(wall_hash(s) for s in range(start,start+n))
    for p in out.parent.glob('**/arena_config.json'):excluded.update(json.loads(p.read_text())['wall_ids'])
    if len(set(training_walls))!=len(training_walls) or set(training_walls)&excluded:raise ValueError('PPO training wall leakage')
    provenance=dict(config=cfg,bc=identity(checkpoint),frozen={k:identity(v) for k,v in frozen.items()},critic_gradients='detached from actor backbone')
    (out/'training_walls.json').write_text(json.dumps(dict(seed=cfg['seed'],planned_wall_ids=training_walls),indent=2))
    counters=dict(env_steps=0,agent_steps=0,gradient_updates=0,completed_episodes=0);history=[];cursor=0;first=1;elapsed=0.;evaluations=[]
    if resume:
        if json.loads((out/'experiment.json').read_text())!=provenance:raise ValueError('PPO resume identities changed')
        old=torch.load(out/'last.pt',map_location='cpu');policy.load_state_dict(old['policy']);learner.optimizer.load_state_dict(old['optimizer'])
        first=old['iteration']+1;counters=old['counters'];cursor=old['episode_cursor'];history=old['rl_history'];elapsed=old['elapsed_seconds'];evaluations=old['evaluations']
        torch.set_rng_state(old['torch_rng']);np.random.set_state(old['numpy_rng']);random.setstate(old['python_rng'])
        if cfg['device']=='mps' and 'mps_rng' in old:torch.mps.set_rng_state(old['mps_rng'])
    else:
        (out/'experiment.json').write_text(json.dumps(provenance,indent=2));(out/'config.json').write_text(json.dumps(cfg,indent=2))
        shutil.copy2(checkpoint,out/'initial_bc.pt');shutil.copy2(checkpoint,out/'eval_best.pt')
    eval_opponents={'random':None,'teacher':None,**rl['eval_opponents']}
    def evaluate(path,iteration):
        directory=out/('dev_%04d'%iteration)
        result=run_arena({'candidate':str(path)},eval_opponents,'reports/round3/raw/split_manifest.json',directory,
            rl['eval_seed'],rl['eval_walls'],workers=cfg['actors'],purpose='development',resume=directory.exists(),registry_root=out.parent)
        score=result['aggregate']['scores']['candidate'];evaluations.append(dict(iteration=iteration,score=score,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()))
        (out/'evaluations.json').write_text(json.dumps(evaluations,indent=2));return score['mean']
    if not evaluations:evaluate(out/'initial_bc.pt',0)
    best=max(e['score']['mean'] for e in evaluations);best_post=max([e['score']['mean'] for e in evaluations if e['iteration']>rl['warmup_iterations']]+[-float('inf')])
    router=StaticPolicyRouter({str(i):PolicyRef('shared',policy) for i in range(4)})
    worker_factory=partial(factory,config=cfg,opponents=frozen,mode=rl['mode'])
    start=time.monotonic()
    with ParallelEpisodeCollector(worker_factory,router,actors=cfg['actors'],episodes_per_actor=cfg['episodes_per_actor'],
        gamma=cfg['ppo']['gamma'],gae_lambda=cfg['ppo']['gae_lambda'],seed=cfg['seed'],timeout=cfg.get('actor_timeout',600)) as collector:
        collector.episode_cursor=cursor
        workflow=MultiPolicyOnPolicyWorkflow(collector,{'shared':OnPolicyStack(policy,GAEProcessor(cfg['ppo']['gamma'],cfg['ppo']['gae_lambda']),learner)})
        with (out/'metrics.jsonl').open('a' if resume else 'w') as log:
            for iteration in range(first,cfg['iterations']+1):
                warmup=iteration<=rl['warmup_iterations']
                for name,p in policy.module.named_parameters():p.requires_grad_(not warmup or name.startswith('critic.'))
                before={n:p.detach().cpu().clone() for n,p in policy.module.named_parameters() if not n.startswith('critic.')}
                step_start=time.monotonic();metrics=workflow.step()
                actor_changed=any(not torch.equal(before[n],p.detach().cpu()) for n,p in policy.module.named_parameters() if n in before)
                if warmup and actor_changed:raise AssertionError('critic warmup mutated actor')
                if not all(np.isfinite(v) for v in metrics.values()):raise FloatingPointError('PPO nonfinite')
                for key in counters:counters[key]+=int(metrics[key])
                trajectories=workflow.last_collection.trajectories
                rewards=[float(t.data.get('reward').sum()) for t in trajectories]
                # Two learning seats per game; draw indicator is reconstructed
                # from both terminal rewards (both zero iff the game is drawn).
                by_game={}
                for t,r in zip(trajectories,rewards):by_game.setdefault((t.key.env_id,t.key.episode_id),[]).append(r)
                row=dict(iteration=iteration,warmup=warmup,actor_changed=actor_changed,metrics=metrics,counters=dict(counters),
                    seconds=elapsed+time.monotonic()-start,iteration_seconds=time.monotonic()-step_start,
                    learner_terminal_rewards=rewards,mean_terminal_reward=float(np.mean(rewards)),
                    non_draw_rate=float(np.mean([any(r!=0 for r in values) for values in by_game.values()])),
                    episode_seed_start=cfg['seed']+collector.episode_cursor-int(metrics['completed_episodes']),episode_seed_end=cfg['seed']+collector.episode_cursor-1,
                    opponent_mode=rl['mode'],opponent_cycle=sorted(frozen),
                    opponent_game_counts={name:sum(sorted(frozen)[(seed//4)%len(frozen)]==name for seed in range(cfg['seed']+collector.episode_cursor-int(metrics['completed_episodes']),cfg['seed']+collector.episode_cursor)) for name in sorted(frozen)})
                history.append(row);log.write(json.dumps(row)+'\n');log.flush();print(json.dumps({k:row[k] for k in ('iteration','warmup','actor_changed','seconds','mean_terminal_reward','non_draw_rate')}),flush=True)
                state=dict(policy={k:v.detach().cpu() for k,v in policy.state_dict().items()},optimizer=learner.optimizer.state_dict(),config=cfg,
                    iteration=iteration,counters=counters,episode_cursor=collector.episode_cursor,torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                    rl_history=history,elapsed_seconds=elapsed+time.monotonic()-start,evaluations=evaluations)
                if cfg['device']=='mps':state['mps_rng']=torch.mps.get_rng_state()
                atomic_save(state,out/'last.pt')
                if iteration%rl['eval_every']==0 or iteration==cfg['iterations'] or iteration==rl['warmup_iterations']:
                    snap=out/('iteration_%04d.pt'%iteration);atomic_save(state,snap);score=evaluate(snap,iteration)
                    if not warmup and score>best_post:best_post=score;shutil.copy2(snap,out/'ppo_best.pt')
                    if score>best:best=score;shutil.copy2(snap,out/'eval_best.pt')
                    state.update(evaluations=evaluations,elapsed_seconds=elapsed+time.monotonic()-start);atomic_save(state,out/'last.pt')
                    # Stop only after three post-warmup evaluations all below BC
                    # by a predeclared practical margin; not a significance test.
                    recent=[e for e in evaluations if e['iteration']>rl['warmup_iterations']][-3:]
                    if len(recent)==3 and all(e['score']['mean']<evaluations[0]['score']['mean']-rl.get('stop_margin',.04) for e in recent):break
    summary=dict(counters,iteration=iteration,seconds=elapsed+time.monotonic()-start,best_development_score=best,best_post_ppo_score=best_post,
        warmup_episodes=rl['warmup_iterations']*cfg['actors']*cfg['episodes_per_actor'],stopped_early=iteration<cfg['iterations'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    train(a.config,a.checkpoint,a.output,a.resume)

if __name__=='__main__':main()
