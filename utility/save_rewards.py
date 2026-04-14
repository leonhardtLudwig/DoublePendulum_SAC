from utility.reward_analysis import extract_all_runs, plot_eval_reward


runs = {
    "pendubot_sac_and_snes": "./pendubot_models/pendu_sac_and_snes/log/",
    "pendubot_sac_lqr": "./pendubot_models/pendu_sac_lqr/log/",
    "acrobot_sac_and_snes": "./acrobot_models/acro_sac_and_snes/log/",
    "acrobot_sac_lqr": "./acrobot_models/acro_sac_lqr/log/",
}

df = extract_all_runs(runs, tag="mean_eval_reward")
plot_eval_reward(df, smooth_window=5, save_to="./results/eval_reward.png", show=False)
