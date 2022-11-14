from common.argument_parser import GeneralArgumentParser
from exps.run_algos import run_saute, run_simmer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


if __name__ == "__main__":

    exp_parser = GeneralArgumentParser()
    args = exp_parser.parse_args()
    current_experiment = args.experiment 
### Single Pendulum        
    if current_experiment == 10:
        from exps.single_pendulum.sac_cfg import cfg
        run_saute(**cfg, smoketest=args.smoketest)    
    elif current_experiment == 11:
        from exps.single_pendulum.ppo_cfg import cfg
        run_saute(**cfg, smoketest=args.smoketest)    
    elif current_experiment == 12:
        from exps.single_pendulum.trpo_cfg import cfg
        run_saute(**cfg, smoketest=args.smoketest)    
    elif current_experiment == 13:
        from exps.single_pendulum.ablation_cfg import cfg
        run_saute(**cfg, smoketest=args.smoketest)    
    elif current_experiment == 14:
        from exps.single_pendulum.pi_simmer_cfg import cfg
        run_simmer(**cfg, smoketest=args.smoketest)    
    elif current_experiment == 15:
        from exps.single_pendulum.q_simmer_cfg import cfg
        run_simmer(**cfg, smoketest=args.smoketest)    
    elif current_experiment == 16:
        from exps.single_pendulum.key_observation_cfg import cfg
        run_saute(**cfg, smoketest=args.smoketest)    
### Double Pendulum      
    elif current_experiment == 20:
        from exps.double_pendulum.trpo_cfg import cfg
        run_saute(**cfg, smoketest=args.smoketest)    
    elif current_experiment == 21:
        from exps.double_pendulum.trpo_lag_cfg import cfg
        run_saute(**cfg, smoketest=args.smoketest)    
    elif current_experiment == 22:
        from exps.double_pendulum.naive_generalization import run
        run(**cfg, smoketest=args.smoketest)    
    elif current_experiment == 23:
        from exps.double_pendulum.zero_shot_generalization import run
        run(**cfg, smoketest=args.smoketest)    
    elif current_experiment == 24:
        from exps.double_pendulum.ablation_unsafe_cfg import cfg
        run_saute(**cfg, smoketest=args.smoketest)    
    elif current_experiment == 25:
        from exps.double_pendulum.ablation_components_cfg import cfg
        run_saute(**cfg, smoketest=args.smoketest)    
### Reacher
    elif current_experiment == 30:
        from exps.reacher.reacher_cfg import cfg
        run_saute(**cfg, smoketest=args.smoketest)    
### Safety Gym
    elif current_experiment == 40:
        from exps.safety_gym.sg_point_cfg import cfg
        run_saute(**cfg, smoketest=args.smoketest)    
### testing for minor bugs 
    elif current_experiment == -1: # experminetal feature
        args.smoketest = -1
        ## single pendulum            
        from exps.single_pendulum.sac_cfg import cfg
        run_saute(**cfg, smoketest=args.smoketest)    
        from exps.single_pendulum.ppo_cfg import cfg
        run_saute(**cfg, smoketest=args.smoketest)    
        from exps.single_pendulum.trpo_cfg import cfg
        run_saute(**cfg, smoketest=args.smoketest)    
        from exps.single_pendulum.pi_simmer_cfg import cfg
        run_simmer(**cfg, smoketest=args.smoketest)    
        from exps.single_pendulum.q_simmer_cfg import cfg
        run_simmer(**cfg, smoketest=args.smoketest)           
        ## double pendulum 
        from exps.double_pendulum.trpo_cfg import cfg
        run_saute(**cfg, smoketest=args.smoketest)    
        from exps.double_pendulum.trpo_lag_cfg import cfg
        run_saute(**cfg, smoketest=args.smoketest)    
        # furtrher exps
        from exps.reacher.reacher_cfg import cfg
        run_saute(**cfg, smoketest=args.smoketest)    
        from exps.safety_gym.sg_point_cfg import cfg
        run_saute(**cfg, smoketest=args.smoketest)    
    else:
        raise NotImplementedError        
    
   