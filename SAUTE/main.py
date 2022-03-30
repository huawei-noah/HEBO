from common.argument_parser import GeneralArgumentParser


if __name__ == "__main__":

    exp_parser = GeneralArgumentParser()
    args = exp_parser.parse_args()

    current_experiment = args.experiment 
### Mountain Car
    if current_experiment == 0:
        from exps.mountain_car.tf_sac_v1 import run_tf_sac_v1          
        run_tf_sac_v1(num_exps=args.num_exps, smoketest=args.smoketest)    
### Single Pendulum        
    elif current_experiment == 10:
        from exps.single_pendulum.tf_sac_pendulum_v1 import run_tf_sac_pendulum_v1          
        run_tf_sac_pendulum_v1(num_exps=args.num_exps, smoketest=args.smoketest)    
    elif current_experiment == 11:
        from exps.single_pendulum.tf_ppo_pendulum_v1 import run_tf_ppo_pendulum_v1          
        run_tf_ppo_pendulum_v1(num_exps=args.num_exps, smoketest=args.smoketest)    
    elif current_experiment == 12:
        from exps.single_pendulum.tf_trpo_pendulum_v1 import run_tf_trpo_pendulum_v1          
        run_tf_trpo_pendulum_v1(num_exps=args.num_exps, smoketest=args.smoketest)            
    elif current_experiment == 13:
        from exps.single_pendulum.tf_trpo_pendulum_v2_abblation import run_tf_trpo_pendulum_v2_abblation          
        run_tf_trpo_pendulum_v2_abblation(num_exps=args.num_exps, smoketest=args.smoketest)            
### Double Pendulum      
    elif current_experiment == 20:
        from exps.double_pendulum.tf_trpo_double_pendulum_v1 import run_tf_trpo_double_pendulum_v1          
        run_tf_trpo_double_pendulum_v1(num_exps=args.num_exps, smoketest=args.smoketest)    
    elif current_experiment == 21:
        from exps.double_pendulum.tf_trpo_double_pendulum_v2 import run_tf_trpo_double_pendulum_v2
        run_tf_trpo_double_pendulum_v2(num_exps=args.num_exps, smoketest=args.smoketest)
    elif current_experiment == 22:
        from exps.double_pendulum.tf_trpo_double_pendulum_v5 import run_tf_trpo_double_pendulum_v5
        run_tf_trpo_double_pendulum_v5(num_exps=args.num_exps, smoketest=args.smoketest)
    elif current_experiment == 23:
        from exps.double_pendulum.tf_trpo_double_pendulum_v3 import run_tf_trpo_double_pendulum_v3
        run_tf_trpo_double_pendulum_v3(num_exps=args.num_exps, smoketest=args.smoketest)
    elif current_experiment == 24:
        from exps.double_pendulum.tf_trpo_double_pendulum_v4 import run_tf_trpo_double_pendulum_v4
        run_tf_trpo_double_pendulum_v4(num_exps=args.num_exps, smoketest=args.smoketest)
### Reacher
    elif current_experiment == 30:
        from exps.reacher.tf_trpo_reacher_v1 import run_tf_trpo_reacher_v1
        run_tf_trpo_reacher_v1(num_exps=args.num_exps, smoketest=args.smoketest)
### Safety Gym
    elif current_experiment == 40:
        from exps.safety_gym.tf_trpo_safety_gym_v1 import run_tf_trpo_safety_gym_v1
        run_tf_trpo_safety_gym_v1(num_exps=args.num_exps, smoketest=args.smoketest)
### testing for minor bugs 
    elif current_experiment == -1: # experminetal feature
        from exps.single_pendulum.tf_sac_pendulum_v1 import run_tf_sac_pendulum_v1          
        from exps.single_pendulum.tf_ppo_pendulum_v1 import run_tf_ppo_pendulum_v1          
        from exps.single_pendulum.tf_trpo_pendulum_v1 import run_tf_trpo_pendulum_v1          
        from exps.double_pendulum.tf_trpo_double_pendulum_v1 import run_tf_trpo_double_pendulum_v1          
        from exps.double_pendulum.tf_trpo_double_pendulum_v2 import run_tf_trpo_double_pendulum_v2          
        from exps.double_pendulum.tf_trpo_double_pendulum_v3 import run_tf_trpo_double_pendulum_v3          
        from exps.double_pendulum.tf_trpo_double_pendulum_v4 import run_tf_trpo_double_pendulum_v4          
        from exps.reacher.tf_trpo_reacher_v1 import run_tf_trpo_reacher_v1
        from exps.safety_gym.tf_trpo_safety_gym_v1 import run_tf_trpo_safety_gym_v1
        args.smoketest = -1
        ## single pendulum            
        run_tf_sac_pendulum_v1(num_exps=args.num_exps, smoketest=args.smoketest)    
        run_tf_ppo_pendulum_v1(num_exps=args.num_exps, smoketest=args.smoketest)    
        run_tf_trpo_pendulum_v1(num_exps=args.num_exps, smoketest=args.smoketest)            
        ## double pendulum 
        run_tf_trpo_double_pendulum_v1(num_exps=args.num_exps, smoketest=args.smoketest)    
        run_tf_trpo_double_pendulum_v2(num_exps=args.num_exps, smoketest=args.smoketest)    
        run_tf_trpo_double_pendulum_v3(num_exps=args.num_exps, smoketest=args.smoketest)    
        run_tf_trpo_double_pendulum_v4(num_exps=args.num_exps, smoketest=args.smoketest)    
        ## reacher 
        run_tf_trpo_reacher_v1(num_exps=args.num_exps, smoketest=args.smoketest)
        ## safety gym
        run_tf_trpo_safety_gym_v1(num_exps=args.num_exps, smoketest=args.smoketest)
    else:
        raise NotImplementedError        
    
   