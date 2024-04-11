# -*- coding: utf-8 -*-
from __future__ import division
import os
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import torch
import retro

from env2 import Env


# Test DQN
def test(args, T, dqn, dqn2, env0, metrics, results_dir, evaluate=False):
  env = Env(env0)
  env.eval()
  metrics['steps'].append(T)
  T_reward1, T_reward2 = [], []
  reward1_sum, reward2_sum = 0, 0

  # Test performance over several episodes
  done = True
  for _ in range(args.evaluation_episodes):
    while True:
      if done:
        state, reward, done = env.reset().to(args.device), 0, False

      action = dqn.act_e_greedy(state)  # Choose an action ε-greedily
      action2 = dqn2.act_e_greedy(torch.flip(state,[2]))
      state, reward, done, info  = env.step_2P(action, action2)  # Step
      reward1_sum += reward[0]
      reward2_sum += reward[1]
      if args.render:
        env.render()

      if done:
        T_reward1.append(reward1_sum)
        T_reward2.append(reward2_sum)
        reward1_sum, reward2_sum = 0, 0
        break

      state = state.to(args.device)
  # env.close()

  avg_reward1, avg_reward2 = sum(T_reward1) / len(T_reward1), sum(T_reward2) / len(T_reward2)
  if not evaluate:
    # Save model parameters if improved
    if avg_reward1 > metrics['best_avg_reward1']:
      metrics['best_avg_reward1'] = avg_reward1
      dqn.save(results_dir)

    if avg_reward2 > metrics['best_avg_reward2']:
      metrics['best_avg_reward2'] = avg_reward2
      dqn2.save(results_dir)

    # Append to results and save metrics
    metrics['reward1'].append(T_reward1)
    metrics['reward2'].append(T_reward2)

    # Plot
    _plot_line(metrics['steps'], metrics['reward1'], 'Reward-1', path=results_dir)
    _plot_line(metrics['steps'], metrics['reward2'], 'Reward-2', path=results_dir)

  # # Return average reward and Q-value
  # return avg_reward, avg_Q


# Plots min, max and mean + standard deviation bars of a population over time
def _plot_line(xs, ys_population, title, path=''):
  max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

  ys = torch.tensor(ys_population, dtype=torch.float32)
  ys_min, ys_max, ys_mean, ys_std = ys.min(1)[0].squeeze(), ys.max(1)[0].squeeze(), ys.mean(1).squeeze(), ys.std(1).squeeze()
  ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

  trace_max = Scatter(x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash='dash'), name='Max')
  trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
  trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
  trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
  trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash='dash'), name='Min')

  plotly.offline.plot({
    'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
    'layout': dict(title=title, xaxis={'title': 'Step'}, yaxis={'title': title})
  }, filename=os.path.join(path, title + '.html'), auto_open=False)
