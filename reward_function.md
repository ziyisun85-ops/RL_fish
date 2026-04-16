# Reward Function Description

This document describes the reward function currently implemented in the project. The implementation is defined mainly in:

- `envs/fish_env.py`
- `configs/default_config.py`

The description below follows the current code exactly, rather than an abstract design target.

## 1. Overall Reward Formula

At each control step, the environment computes the total reward as:

```text
R_t =
  w_target  * r_target
+ w_obs     * r_obs
+ w_heading * r_heading
+ w_smooth  * r_smooth
- step_penalty
- wall_collision_cost * 1[wall_collision]
- timeout_penalty     * 1[timeout]
+ success_reward      * 1[reached_goal]
```

In the current default configuration:

```text
w_target  = 85.0
w_obs     = 14.0
w_heading = 2.0
w_smooth  = 1.0

step_penalty        = 0.01
wall_collision_cost = 12.0
timeout_penalty     = 8.0
success_reward      = 60.0
```

Therefore, the concrete reward expression is:

```text
R_t =
  85.0 * r_target
+ 14.0 * r_obs
+  2.0 * r_heading
+  1.0 * r_smooth
-  0.01
- 12.0 * 1[wall_collision]
-  8.0 * 1[timeout]
+ 60.0 * 1[reached_goal]
```

## 2. Target Progress Reward

The target-progress term rewards the fish for getting closer to the goal:

```text
r_target = target_progress_scale * (d_prev_norm - d_curr_norm)
```

with:

```text
target_progress_scale = 1.0
```

The normalized goal distance is:

```text
d_norm = clip(goal_distance / initial_goal_distance, 0, goal_distance_clip)
```

where:

```text
goal_distance_clip = 1.0
```

Interpretation:

- If the fish moves closer to the goal between two consecutive control steps, `r_target > 0`.
- If the fish moves away from the goal, `r_target < 0`.
- The distance term is normalized by the initial goal distance of the current episode.

## 3. Obstacle Avoidance Reward

The obstacle term is actually a penalty term. The environment first computes the current obstacle distance:

```text
obstacle_distance = min(
    min_obstacle_clearance,
    local_obstacle_obs.edge_distance if available
)
```

Then the reward uses a piecewise definition:

### 3.1 Far from obstacles

If:

```text
obstacle_distance >= obstacle_buffer_distance
```

then:

```text
r_obs = 0
```

### 3.2 Very close to obstacles

If:

```text
obstacle_distance <= obstacle_danger_distance
```

then:

```text
r_obs = -obstacle_collision_penalty
```

### 3.3 In the outer buffer region

If:

```text
obstacle_safe_distance < obstacle_distance < obstacle_buffer_distance
```

then:

```text
normalized_buffer_gap =
    (obstacle_buffer_distance - obstacle_distance)
    / (obstacle_buffer_distance - obstacle_safe_distance)

r_obs = -obstacle_buffer_penalty * normalized_buffer_gap
```

### 3.4 In the inner danger-to-safe region

If:

```text
obstacle_danger_distance < obstacle_distance <= obstacle_safe_distance
```

then:

```text
normalized_gap =
    (obstacle_safe_distance - obstacle_distance)
    / (obstacle_safe_distance - obstacle_danger_distance)

r_obs = -(normalized_gap^2)
```

Current default parameters:

```text
obstacle_safe_distance     = 0.70
obstacle_buffer_distance   = 0.95
obstacle_danger_distance   = 0.08
obstacle_collision_penalty = 1.0
obstacle_buffer_penalty    = 0.12
```

After weighting in the total reward:

```text
weighted obstacle term = 14.0 * r_obs
```

This means the current implementation adds:

- no obstacle penalty when sufficiently far away,
- a small penalty in the outer buffer zone,
- a stronger nonlinear penalty when entering the inner safety zone,
- and the strongest fixed penalty inside the danger zone.

## 4. Heading Reward

The heading term penalizes angular misalignment with the goal direction.

First compute the desired yaw:

```text
desired_yaw = atan2(goal_y - y, goal_x - x)
```

Then compute heading error:

```text
heading_error = wrap_to_pi(desired_yaw - yaw)
normalized_heading_error = |heading_error| / pi
```

The reward term is:

```text
r_heading = -heading_error_scale * normalized_heading_error
```

Current default parameter:

```text
heading_error_scale = 1.0
```

After weighting in the total reward:

```text
weighted heading term = 2.0 * r_heading
```

Interpretation:

- perfect alignment gives a value close to `0`,
- larger heading error gives a more negative value,
- the term is always non-positive.

## 5. Smoothness Reward

The smoothness term penalizes both large actions and abrupt action changes.

Let:

```text
action_magnitude_penalty = current_action^2
action_delta_penalty     = (current_action - prev_action)^2
```

Then:

```text
r_smooth = -(
    smooth_action_l2_scale    * action_magnitude_penalty
  + smooth_action_delta_scale * action_delta_penalty
)
```

Current default parameters:

```text
smooth_action_l2_scale    = 0.12
smooth_action_delta_scale = 0.20
```

Interpretation:

- large steering commands are penalized,
- rapidly changing steering commands are also penalized,
- this term is always non-positive.

## 6. Per-Step and Terminal-Related Terms

### 6.1 Step penalty

Every control step gets:

```text
-step_penalty
```

Current value:

```text
step_penalty = 0.01
```

This encourages shorter and more efficient trajectories.

### 6.2 Wall collision penalty

If the fish is in wall contact at the current step:

```text
-wall_collision_cost
```

Current value:

```text
wall_collision_cost = 12.0
```

### 6.3 Timeout penalty

If the episode is truncated due to timeout, the current step gets:

```text
-timeout_penalty
```

Current value:

```text
timeout_penalty = 8.0
```

Important detail:

- This timeout penalty is applied only on the truncation step.

### 6.4 Success reward

If the head has touched the goal region:

```text
+success_reward
```

Current value:

```text
success_reward = 60.0
```

Important detail:

- In the current implementation, once `reached_goal == True`, this success reward is added on each subsequent step until the post-goal window finishes.
- So this is not a one-time impulse reward; it is a repeated per-step reward during the post-goal continuation period.

## 7. Success and Termination Logic

### 7.1 Goal reached condition

The current success detection is based on head contact with the goal region:

```text
goal_contact = touches_goal_region(head_geom, goal_region)
```

Once contact occurs:

```text
reached_goal = True
goal_reached_step = current_step
```

### 7.2 Delayed termination after reaching the goal

The episode does not terminate immediately when the head first touches the goal.
Instead, the environment keeps running for a post-goal duration:

```text
post_goal_duration_sec = 5.0
```

The episode terminates successfully only after this post-goal duration has elapsed.

### 7.3 Persistent contact failure

If the fish keeps colliding with an obstacle or wall for too many consecutive control steps, the episode terminates as failure.

Current threshold:

```text
persistent_contact_termination_steps = 20
```

The logic is:

- if `reached_goal == True`, persistent-contact counting is reset,
- else if obstacle collision or wall collision is active, the counter increases,
- else the counter resets to zero,
- when the counter reaches `20`, termination reason becomes `persistent_contact_failure`.

Important detail:

- There is no extra standalone terminal penalty specifically for `persistent_contact_failure`.
- Its effect comes indirectly from obstacle penalties, wall penalties, step penalty, and earlier episode termination.

### 7.4 Timeout

If:

```text
elapsed_steps >= max_episode_steps
```

then the episode truncates with timeout.

Current value:

```text
max_episode_steps = 5000
```

However, once `reached_goal == True`, timeout is disabled so that the post-goal window can finish cleanly.

## 8. Reward Terms Logged in `info`

The environment stores the following reward-related fields in `info`:

- `target_reward`
- `obstacle_reward`
- `heading_reward`
- `smooth_reward`
- `step_penalty`
- `wall_collision_cost`
- `timeout_penalty`
- `success_reward`
- `obstacle_distance`
- `heading_alignment_error`

This is useful for debugging reward balance and diagnosing reward hacking or failure modes.

## 9. Practical Interpretation

The current reward design encourages the agent to:

- move toward the goal,
- avoid obstacle proximity,
- keep the heading generally aligned with the goal direction,
- avoid abrupt or unnecessarily large steering actions,
- avoid wasting time,
- avoid wall contact,
- and strongly favor successful goal contact.

At the same time, because `success_reward` is repeatedly added during the 5-second post-goal continuation window, successful episodes can accumulate a substantial positive terminal-phase reward.

## 10. Current Implementation Summary

In short, the current reward is:

```text
Reward =
  progress-to-goal reward
+ obstacle-avoidance penalty
+ heading-alignment penalty
+ smooth-control penalty
+ success bonus
- step cost
- wall-hit cost
- timeout cost
```

with the exact current numeric weights:

```text
85 * progress
+ 14 * obstacle
+ 2 * heading
+ 1 * smooth
- 0.01 per step
- 12 on wall-collision steps
- 8 on timeout step
+ 60 on each step after goal contact until post-goal termination
```
