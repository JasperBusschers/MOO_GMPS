# This is a sample Python script.
from env_base.dam import DamEnv
from maml_examples.maml_experiment_vars import MOD_FUNC
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.algos.maml_il import MAMLIL
from sandbox.rocky.tf.algos.maml_npo import MAMLNPO
from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO
from sandbox.rocky.tf.algos.maml_vpg import MAMLVPG
from rllab.envs.normalized_env import normalize

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from sandbox.rocky.tf.envs.base import TfEnv
import argparse
import tensorflow as tf
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy



def arguments():
    parse = argparse.ArgumentParser()
    #parameters for environment
    parse.add_argument('--env', type=str, default="dam",help='environment to use')
    parse.add_argument('--log_dir', type=str, default="/logs/dam", help='environment to use')
    parse.add_argument('--save_path', type=str, default="C:\\Users\\JasperBusschers\\PycharmProjects\\MOO_GMPS\\logs\\dam12\\run1\\", help='path to store results')
    parse.add_argument('--dagger', type=str, default=None)
    parse.add_argument('--expert_policy_loc', type=str, default=None)
    parse.add_argument('--EXPERT_TRAJ_LOCATION', type=str, default="C:\\Users\\JasperBusschers\\PycharmProjects"
                                                                   "\\MOO_GMPS\\moo_envs\\expert_traj\\dam12\\")
    parse.add_argument('--load_policy', type=str, default=None)
    parse.add_argument('--max_path_length', type=int, default=5)
    parse.add_argument('--seed', type=int, default=1)
    parse.add_argument('--init_flr', type=float, default=0.005)
    parse.add_argument('--fbs', type=int, default=5, help= "fast batch size")
    parse.add_argument('--mbs', type=int, default=5 , help= "number of meta tasks")
    parse.add_argument('--n_parallel', type=int, default=1)
    parse.add_argument('--ldim', type=int, default=4, help='latent dimension')
    parse.add_argument('--expl',   default=False, type=lambda x: (str(x).lower() == 'true'), help='')
    parse.add_argument('--use_corr_term', default=False, type=lambda x: (str(x).lower() == 'true'), help='')
    parse.add_argument('--l2loss_std_mult', type=float, default=0.1)
    parse.add_argument('--extra_input_dim', type=int, default=0)
    parse.add_argument('--extra_input', type=int, default=None)
    parse.add_argument('--beta_steps', type=int, default=1)
    parse.add_argument('--meta_step_size', type=float, default=0.5)
    parse.add_argument('--num_grad_updates', type=int, default=15)
    parse.add_argument('--pre_std_modifier', type=float, default=1.)
    parse.add_argument('--post_std_modifier', type=float, default=0.00001)
    parse.add_argument('--limit_demos_num', type=int, default=None)
    parse.add_argument('--adamSteps', type=int, default=10)
    parse.add_argument('--test_on_training_goals', default=False, type=lambda x: (str(x).lower() == 'true'))
    parse.add_argument('--use_maesn', default=False, type=lambda x: (str(x).lower() == 'true'))
    args = parse.parse_args()
    return args




def run_experiment(args):
    if args.env == "dam":
        env = TfEnv(normalize(DamEnv()))


    policy =  MAMLGaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        grad_step_size=args.init_flr,
        hidden_nonlinearity=tf.nn.relu,
        hidden_sizes=(20, 20),
    )
    baseline = LinearFeatureBaseline(env_spec = env.spec)

    algo = MAMLIL(
        env=env,
        policy=policy,
        load_policy=args.load_policy,
        baseline=baseline,
        save_path=args.save_path,
        batch_size=args.fbs,  # number of trajs for alpha grad update
        max_path_length=args.max_path_length,
        meta_batch_size=args.mbs,  # number of tasks sampled for beta grad update
        num_grad_updates=args.num_grad_updates,  # number of alpha grad updates
        n_itr=100,
        make_video=False,
        use_maml=True,
        use_pooled_goals=True,
        use_corr_term=args.use_corr_term,
        test_on_training_goals=args.test_on_training_goals,
        metalearn_baseline=False,
        limit_demos_num=args.limit_demos_num,
        test_goals_mult=1,
        step_size=args.meta_step_size,
        plot=False,
        beta_steps=args.beta_steps,
        adam_curve=None,
        adam_steps=args.adamSteps,
        pre_std_modifier=args.pre_std_modifier,
        l2loss_std_mult=args.l2loss_std_mult,
        importance_sampling_modifier=MOD_FUNC[''],
        post_std_modifier=args.post_std_modifier,
        expert_trajs_dir=args.EXPERT_TRAJ_LOCATION,
        expert_trajs_suffix='',
        seed=args.seed,
        extra_input=args.extra_input,
        extra_input_dim=(0 if args.extra_input is "" else args.extra_input_dim),
        plotDirPrefix=None,
        latent_dim=args.ldim,
        dagger=args.dagger,
        expert_policy_loc=args.expert_policy_loc
    )
    algo.train()

if __name__ == '__main__':
    args = arguments()
    run_experiment(args)

