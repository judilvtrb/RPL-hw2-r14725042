# HW2: Dexterous in-hand reorientation on UniLab

Three problems on a simulated Allegro hand that holds an object palm-up and
rotates it to goal orientations:

1. **SO(3) reorientation** of a ball, a cube and a cylinder, one policy per
   object, trained with PPO. Well defined: one reward design and one PPO run
   per object.
2. **One policy for three objects.** The policy never learns which object it
   holds and is graded on each object separately. The method is yours: PPO
   is the starting point, not the ceiling, and your report is graded on what
   you tried and why.
3. **Adaptation to hidden physics.** Same object, but its mass, friction and
   centre of mass and the hand's motor gain change every episode, hidden from
   the policy, and grading uses a wider range than training. Build on the
   idea of Rapid Motor Adaptation (Kumar et al., 2021) and the literature you
   find yourself.

Everything is a UniLab task package: it plugs into an unmodified UniLab 1.5.x
checkout through the `unilab.tasks` entry point, and you never edit UniLab
itself. A CPU laptop is enough for every graded cell.

| | Graded cells | Score |
|---|---|---|
| Problem 1 | `so3_absolute/ball`, `so3_absolute/cube`, `so3_absolute/cylinder` | goals per episode, tiers 1 / 2 / 3 per cell |
| Problem 2 | `multi/yaw_absolute` | worst object's goals per episode, tiers 1 / 2 / 3 |
| Problem 3 | `adapt/yaw_absolute/ball` | goals per episode under held-out physics, tiers 1 / 2 / 3 |

Grading:

Basic Challenge is 80% and the Advanced Challenge is 20%.

| Component | Weight |
|---|---|
| Problem 1: single-object reorientation (section 3) | 25% |
| Problem 2: multi-object generalization (section 4) | 20% |
| Problem 3: rapid motor adaptation (section 5) | 20% |
| Report and reproducibility (section 6) | 10% |
| Video demonstration: GIFs and interactive YouTube video (section 7) | 5% |
| Advanced Challenge: imitation learning (section 6.3) | 20% |

A graded cell earns 1 / 2 / 3 points for tiers 1 / 2 / 3, and 0 when the
evaluator cannot run it. Within a Problem the points scale linearly to its
weight: Problem 1 has 9 points over its three cells; Problem 2 has 3 points on its
one cell; Problem 3 has 3 points on the public cell and 3 on the hidden
scenarios. `hw2 grade` prints the cell points; the
report and the video are graded by the TAs from the archive you submit.

## 1. Setup

HW2 is a UniLab *task package*. It is not installed on its own: it goes into
the virtual environment of a UniLab checkout, where UniLab discovers it through
the `unilab.tasks` entry point. Keep the two checkouts side by side; UniLab's
`.venv` is the only Python environment you use.

```
<workspace>/
  UniLab/    UniLab 1.5.x checkout; `make sync` creates UniLab/.venv
  HW2/       this repository, installed into UniLab/.venv as an editable package
```

### 1.1 Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # macOS / Linux
```

uv downloads a matching Python (3.10 to 3.13) by itself; no conda or system
Python is needed. Open a new shell afterwards so `uv` is on your `PATH`.

### 1.2 Install UniLab with the MuJoCo backend

```bash
git clone --branch v1.5.2 https://github.com/Motphys/UniLab.git
cd UniLab
make sync        # = uv sync --extra mujoco; creates UniLab/.venv with torch, MuJoCo and UniLab
cd ..
```

v1.5.2 is the one the grader runs. UniLab's own
installation guide is at
<https://unilabsim.github.io/UniLab-doc/en/1-getting_started/2-installation.html>.
Do not edit anything under `UniLab/`: the grader uses an unmodified checkout,
so a change there is not part of your submission.

### 1.3 Install HW2 into that environment

```bash
git clone <hw2 repo url> HW2
cd HW2
make install     # = cd ../UniLab && uv pip install -e <HW2>
```

This installs `hw2_dexhand` as an *editable* package into `UniLab/.venv`:
edits under `src/hw2_dexhand/`, including your code under `student/`, take
effect without reinstalling. If UniLab is not at `../UniLab`, say where it is:
`make install UNILAB_DIR=/path/to/UniLab`.

### 1.4 Run everything through UniLab's environment

Every command in this file runs from the `HW2` directory through UniLab's
environment:

```bash
uv run --project ../UniLab --no-sync hw2 cells
```

`--project ../UniLab` selects UniLab's `.venv`; `--no-sync` stops uv from
re-syncing that environment, which would remove the HW2 install. If your shell
exports another `VIRTUAL_ENV` (conda, another venv), prefix commands with
`env -u VIRTUAL_ENV`. Define

```bash
alias hw2='env -u VIRTUAL_ENV uv run --project ../UniLab --no-sync hw2'
```

and the rest of this file reads literally.

### 1.5 Generate the grasp caches

```bash
hw2 make-caches      # ball, cube and cylinder; a few minutes per object, once
```

This writes `src/hw2_dexhand/assets/caches/allegro_<object>_grasp_10k.npy`,
the stable starting grasps every episode resets from. The caches are not
shipped and are ignored by git.

### 1.6 Check the install

`hw2 cells` lists every cell the evaluator knows and `hw2 template` prints the
overlay template. The first `hw2 train` builds the MuJoCo scene and tells you
if the grasp caches are missing.

If `hw2` is not found, or `uv run --project ../UniLab --no-sync python -c
"import hw2_dexhand"` fails, the HW2 install was dropped from UniLab's
environment (for example by running `uv sync` or `make sync` in `UniLab`
again). Run `make install` in `HW2` once more.

## 2. How every problem works

### The environment

An Allegro hand (16 position-controlled joints) holds a ball, a cube or a
cylinder palm-up. Each episode starts from a stable grasp drawn from a cache,
lasts 20 seconds (400 control steps at 20 Hz) and ends early if the object
drops. A goal orientation is drawn; when the object is within 0.4 rad of it
the goal counts as reached and a new one is drawn. The score of a policy is
**goals reached per episode**, averaged over 64 fresh episodes.

```
hw2 cells          # every cell the evaluator knows, graded or not
hw2 list           # the packaged owner configs behind them
```

Goal tasks: `yaw_absolute` and `yaw_relative` rotate about the world z axis
(goal in the world frame, or relative to the orientation at reset);
`so3_absolute` and `so3_relative` draw any orientation. Only the cells in the
table above earn points; the others evaluate the same way and are there for
practice and for your own experiments.

### What is fixed and what you decide

| Fixed by the evaluator | Yours |
|---|---|
| hand and object models, scene | observation terms and their history lengths |
| goal distribution, success tolerance 0.4 rad, 20 s episodes | action scale |
| drop termination, grasp-cache resets | reward terms and weights |
| on problem 3: the training randomization range and what the actor may observe | training-time randomization (problems 1 and 2; training only) |
| on problem 2: the policy does not observe which object it holds | network, algorithm, budget, training procedure |

Everything in the right column lives in a YAML *overlay* that is merged on
top of the cell's packaged configuration, so you list only what you change.
At grading the evaluator composes your overlay, re-imposes everything in the
left column and reports what it overrode. Grading always runs the packaged
events: randomization events you add apply to training only, and a critic
observation group you add is not built, since only the actor is graded.

### Workflow

```bash
hw2 template > my_config.yaml                      # fill every ???
hw2 train so3_absolute/ball --config my_config.yaml
hw2 train so3_absolute/ball --config my_config.yaml --cfg   # print the composed config, no training
hw2 play  so3_absolute/ball --config my_config.yaml algo.load_run=-1   # replay the latest run
hw2 eval  so3_absolute/ball logs/rsl_rl_ppo/AllegroInhandReorient/<run> --config my_config.yaml
```

The template lists the reference reward terms, the PPO settings and the
network with their values blanked; `hw2 train` refuses a file with a `???`
left in it. Any other key you add is merged too: a new observation term, a
term dropped with `null`, a new reward term, a different `algo.*`, a
command-line override after the cell name. Training writes
`logs/rsl_rl_ppo/<Task>/<run>/model_*.pt` and `run_config.json`.


### What the evaluator loads

The evaluator builds the actor from your overlay (`algo.actor`, the
observation terms, `algo.obs_groups`), loads the actor's weights alone from
your checkpoint, and runs it deterministically. Two consequences:

* **The checkpoint is a plain rsl_rl checkpoint**, as written by `hw2 train`:
  a torch file whose `model_state_dict` holds the actor's parameters under
  rsl_rl's names, plus the small `unilab_logger_state` entry UniLab's loader
  expects. If you train with your own procedure (problem 2 and problem 3 allow it),
  save in that format; open a checkpoint written by `hw2 train` to see it.
* **`algo.actor.class_name` names the actor class** by dotted path:
  rsl_rl's `MLPModel` or a class of your own under `src/hw2_dexhand/student/`.
  rsl_rl constructs it the way it constructs `MLPModel` and passes every
  other key under `algo.actor` as a keyword argument. The grader runs your
  package, not your working directory, so code the actor needs must live
  under `src/hw2_dexhand/student/`.

A checkpoint whose observation width does not match the overlay scores 0 with
an explanation, so run `hw2 eval` on every graded cell before you submit.

## 3. Problem 1: SO(3) reorientation with PPO

Cells `so3_absolute/ball`, `so3_absolute/cube`, `so3_absolute/cylinder`. The
goal is any orientation, in the world frame. The policy observes the hand's
joint positions and position targets and the object position, each with a
three-frame history, plus the rotation from the current orientation to the
goal; for the cube and cylinder it also observes the object orientation
(current frame only).

What to do:

1. `hw2 template > my_config.yaml` and fill it. Finding a reward design and
   training settings that solve the task is part of the exercise.
2. Train one policy per object. Watch `Metrics/goal/goals_reached` and the
   episode length in the training log.
3. `hw2 eval` each run with the overlay it was trained with. The evaluator's
   number is the graded one; the training metric is an average at reset
   and differs from it.
4. Yaw cells (`yaw_absolute/<object>`) are easier and not graded; use them
   to debug a configuration before paying for SO(3).

The objects differ a lot in difficulty, and the ball is the easiest. On the
cube and the cylinder a policy that holds the object still and collects the
dense orientation term is a strong local optimum: the goal count can sit at
zero for several hundred iterations before it takes off, so a flat curve early
on is not a broken setup. Plan for more than the default budget on those two
cells, or for a reward that leaves that optimum sooner.

Scoring is by the tier of goals per episode. Thresholds are in
`src/hw2_dexhand/eval/rubric.yaml`.

| cell | tiers 1 / 2 / 3 |
|---|---|
| `so3_absolute/ball` | 1.5 / 3.0 / 5.0 |
| `so3_absolute/cube` | 0.25 / 0.75 / 1.5 |
| `so3_absolute/cylinder` | 0.25 / 0.75 / 1.5 |

Standard error at 64 episodes is a few tenths of a goal per episode at low
scores and up to a goal per episode at high ones, so leave a margin above a
threshold.

## 4. Problem 2: one policy for three objects

> **Observation restriction.** The evaluated actor must **not** observe object
> identity, directly or under another name (for example the object orientation
> term or any quantity that reveals which object it holds). A violating
> submission scores 0 on this problem.

Cell `multi/yaw_absolute`. One training run on a pool where every environment
holds the ball, the cube or the cylinder (one third each) and resets it from
that object's grasp cache. The policy never observes which object it holds:
the observation is the single-object layout without the object orientation,
so the same checkpoint runs unchanged on each single-object scene at grading.

### How it is scored

The evaluator runs the checkpoint on the ball, the cube and the cylinder
scenes separately, 64 episodes each, and the cell's score is the **worst** of
the three. One strong object cannot carry the grade. Points come from the
`multi_tiers` table in `rubric.yaml`: a worst object at 0.5 / 1.0 / 2.0 goals
per episode reaches tiers 1 / 2 / 3. The three per-object results are
reported next to the score so you can see which object limits you.

The problem's 20% comes from this cell. Your report section for the problem
(section 6) is graded on the method (what you built beyond the packaged PPO
recipe, why you expected it to help on this problem, and how original it is) and
on the analysis (controlled comparison against the packaged recipe at the
same budget, per-object numbers, what failed). `hw2 train multi/yaw_absolute`
with no `--config` runs the packaged recipe. The pooled
`Metrics/goal/goals_reached` curve printed during training averages over all
objects at reset; it is not the graded number.

### What is open

The graded object is the actor and nothing else. How you obtain it is yours:
a different algorithm, a different network, a curriculum over objects,
auxiliary losses, information that only the training side sees, separate
training stages, your own training loop. Three constraints: the actor
observes only sensor-like quantities and never which object it holds (at
grading the evaluator drops the packaged object-orientation term, refuses an
overlay whose actor group reads the object orientation under any other name or
uses an observation term that is not one of the packaged terms, so that the
cell scores 0 with the reason, and your code is read); the
checkpoint loads as described in section 2; and the comparison in your report
is against the packaged recipe at the same sample budget, so the method, not
the compute, is what you are showing. `multi/so3_absolute` evaluates the same
way and has no calibrated thresholds; results there belong in the report,
not the score.

## 5. Problem 3: adapting to hidden physics

### The problem

> **Warning: do not train on the held-out range.** Training on the held-out
> parameter ranges is strictly prohibited. Train only on the training range
> below. The evaluated actor also cannot directly access the hidden physical
> parameters (mass, friction, centre of mass, motor gain). Violations score 0
> on this problem.

Cell `adapt/yaw_absolute/ball`. At every reset the simulator redraws four
properties and hides them from the policy: the ball's mass, its surface
friction, the position of its centre of mass, and the position gain
(stiffness) of the hand's sixteen motors. Same hand, same object, same goals
as the yaw cell of problem 1; only the physics changes between episodes.
Training draws the four properties from the *training range*. Grading draws
them from a wider *held-out range* that extends past the training range on
every side, so a policy that treats every episode the same way may degrade outside the
training range.

| Property | Training range | Held-out range (grading) |
|---|---|---|
| object mass | x0.5 to x1.5 | x0.3 to x2.0 |
| object friction | x0.6 to x1.4 | x0.4 to x1.8 |
| centre-of-mass offset, each axis | -0.01 to +0.01 m | -0.02 to +0.02 m |
| motor position gain | x0.7 to x1.3 | x0.5 to x1.6 |

The starting point is Rapid Motor Adaptation:

> A. Kumar, Z. Fu, D. Pathak, J. Malik. *RMA: Rapid Motor Adaptation for
> Legged Robots.* Robotics: Science and Systems (RSS), 2021.
> arXiv:2107.04034.

The paper trains a walking policy that adapts online to terrain and body
changes it cannot observe, using only its own proprioceptive history, and
describes how such a policy can be trained in simulation when the hidden
quantities are available there. Read it, decide what carries over to an
in-hand task where the hidden quantities are the ones in the table, and
survey the work that followed it yourselves: the report should say what you
read, what you took from it and what you changed.

### What is fixed

Everything problem 1 fixes, plus:

* **The training range.** The two randomization events of the packaged
  `allegro_adapt` owner are evaluator-owned at training time as well:
  `hw2 train adapt/...` refuses an overlay or command-line override that
  changes `env.events.randomize_object_physics` or
  `env.events.randomize_hand_gains`. Do not widen the range by other means
  (your own training loop, a copied owner); submitted code is read, and a
  policy trained on the held-out range scores 0 on the problem.
* **What the graded actor sees.** Only the `actor` observation group: the
  policy terms you declare (by default joint positions, joint targets and the
  object position with a 3-frame history, plus the goal rotation). You may
  add terms, drop terms and change history lengths, choosing from the packaged
  observation terms in `hw2_dexhand.tasks.allegro_inhand.terms` and UniLab's
  `unilab.envs.mdp`: they read the hand's joint state, the object's pose and
  velocities, sensors and the goal, and none of them returns the sampled
  values or the simulator's mass, friction, inertia or gain parameters.
  Anything else the actor should compute from those inputs belongs in your
  actor class. At grading the evaluator removes the two sampled-value terms
  from the environment and refuses a submission whose actor group lists
  `critic`, whose non-critic observation terms return the sampled values, or
  whose non-critic terms come from any other module; the cell then scores 0
  with the reason in the report.
* **What training may see.** The `critic` group is the policy terms followed
  by the six sampled values (mass scale, friction scale, centre-of-mass offset
  x, y, z, gain scale); a critic term you add in an overlay is appended after
  them. They may feed the critic or anything else you train; only the
  submitted actor must do without them.
* **Evaluation.** 64 fresh episodes from the grasp cache, 16 environments in
  parallel, deterministic actions. The public numbers use seed 0; the hidden
  scenarios use another seed.

`hw2 template --adapt` prints the overlay with the keys this contract
involves; `hw2 train adapt/yaw_absolute/ball --config adapt_config.yaml`
trains on the fixed range.

### Public test cases

```bash
hw2 eval adapt/yaw_absolute/ball <run> --config adapt_config.yaml --challenge public
```

The first line is the graded cell: all four properties drawn from the
held-out range. Then eight scenarios, each moving **one** property outside the
training range while the other three stay inside it. They tell you which
property your policy copes with. Goals per episode, 64 episodes, seed 0.

| Scenario | What changes |
|---|---|
| graded cell | all four properties on the held-out range |
| `in_range` | nothing: all four inside the training range (control) |
| `light` | mass x0.3 to x0.5 |
| `heavy` | mass x1.5 to x2.0 |
| `slippery` | friction x0.4 to x0.6 |
| `sticky` | friction x1.4 to x1.8 |
| `off_center` | centre-of-mass offset up to 0.02 m per axis |
| `weak_motor` | motor gain x0.5 to x0.7 |
| `strong_motor` | motor gain x1.3 to x1.6 |

The graded cell uses the tiers in `rubric.yaml`: 5.0 / 6.5 / 8.0 goals per
episode for tiers 1 / 2 / 3.

### Hidden test cases

Same object, same envelope, another seed. Single properties pinned at the
extremes of the held-out range with the others drawn from it, and
combinations of several properties at once. Each hidden scenario is scored
against two reference policies measured under that same scenario: one trained
on the fixed range that ignores the physics, and one that reads the sampled
values during evaluation. Your normalised score on a scenario is
`clip((yours - ignores) / (reads - ignores), 0, 1)`, and the hidden
scenarios' normalised scores are averaged for 3 points. The public cell's 3
points and the hidden scenarios' 3 points together make the problem's 20%.

### What you build

Anything that satisfies the contract: an actor class the evaluator can
construct from your overlay, and a checkpoint it can load. How you train it
is yours: extra training stages, your own scripts, extra losses, anything
that respects the fixed training range. Use the public scenarios to find out
which properties your policy handles and which it does not; your report
section for the problem (section 6) is graded on that analysis and on what your
policy actually infers about the physics it is given.

## 6. Report and reproducibility (10%)

### 6.1 `report.md`

Submit a Markdown report named exactly `report.md` in the repository root
(not a PDF). It must include:

1. **Training dynamics and hyperparameter analysis**
   * How do you compare hyperparameter sweeps?
   * How do you choose the network architecture and training iterations?
   * How do you design reward functions?
   * How do you maximize training efficiency on your device?
2. **Failure modes and limitations**, covering
   * the policy holds the object without rotating it;
   * the object is dropped during manipulation;
   * the policy performs well on one object but fails on others;
   * the policy degrades under unseen physics;
   * training reward increases without improving task success.

   For each: why does it occur, how did you diagnose it, what did you try to
   resolve it, and what limitations remain?
3. **Demo videos** (section 7): the `.gif` files for all problems, embedded.
4. **Interactive video** (section 7): the YouTube link.
5. **Advanced Challenge** write-up (section 6.3).

### 6.2 `AI_history.md`

Summarize and export your conversation history with AI agents to
`AI_history.md` in the repository root.

### 6.3 Advanced Challenge: imitation learning (20%)

* Implement an imitation learning approach to train a dexterous in-hand
  manipulation policy from demonstration data (for example casually captured
  videos or expert trajectories). Any imitation learning algorithm is allowed.
* Evaluate the policy on the provided reorientation tasks and compare its
  performance with the PPO baseline.
* Describe the methodology, training process and experimental results in
  `report.md`.
* Provide a video demonstration of the trained policy in `report.md`.


## 7. Video demonstration (5%)

Two kinds of video go into `report.md`.

### 7.1 Demo videos (`.gif`)

Export demonstration videos for **all problems** as `.gif` files and embed
them in `report.md`: one clip per problem 1 cell, one clip of the problem 2 policy
on the mixed pool (every third environment holds a different object), and one
clip of the problem 3 policy. Label each clip with the cell and the run it shows,
and say in the report which checkpoints appear.

### 7.2 Interactive video (YouTube)

Record a video in which you **manually control the goal orientation with the
arrow keys** and change the target orientation to demonstrate the policy's
goal-reaching ability. Upload it to **YouTube** and put the link in
`report.md`. State in the report how the goal was controlled by hand.

### 7.3 Rendering clips

`hw2 play` renders a clip of a run to `<run dir>/play_video.mp4`; convert it to
`.gif` for the report:

```bash
hw2 play so3_absolute/ball --config my_config.yaml algo.load_run=-1 training.play_render_mode=record
hw2 play multi/yaw_absolute --config multi_config.yaml algo.load_run=-1 training.play_render_mode=record
hw2 play adapt/yaw_absolute/ball --config adapt_config.yaml algo.load_run=-1 training.play_render_mode=record
```

`algo.load_run=-1` replays the latest run of that task; give a run directory
or a `model_*.pt` path instead to pick another. `training.play_env_num`
(default 16) sets how many environments appear and `training.play_steps`
(default 200 control steps, 10 s) the length of the clip. The goal frame is
drawn above each object. Convert and join the clips with any video tool.

## 8. Submit

### 8.1 GitHub and NTU COOL

* Create a repository from the `RPL-hw2-Dexterous-In-Hand-Reorientation`
  template and name it `RPL-hw2-{your_student_id}`.
* Submit your code, `report.md` and `AI_history.md` in it. The TAs clone the
  **main** branch, so everything must be on `main`.
* Also commit what the grader needs: `submission.yaml`, the overlay configs,
  the checkpoints it names, and `src/hw2_dexhand/student/` if you wrote code.
* Submit your GitHub repository link to **NTU COOL Homework 2**.
* Deadline: **2026/11/30 (Mon.) 23:59 (GMT+8)**. The score is multiplied by
  0.9 for each additional day of delay; no grace day is granted.
* Incorrect filenames or failure to complete every submission step result in
  no credit.

### 8.2 `submission.yaml`

```yaml
# submission.yaml
student_id: your_student_id
config: my_config.yaml                 # overlay used for every cell (optional)
cells:
  so3_absolute/ball: runs/so3_ball     # run dir -> highest model_*.pt
  so3_absolute/cube: runs/so3_cube
  so3_absolute/cylinder: runs/so3_cylinder
  multi/yaw_absolute:
    checkpoint: runs/multi/model_399.pt
    config: multi_config.yaml          # per-cell overlay (optional)
  adapt/yaw_absolute/ball:
    checkpoint: runs/adapt_ball
    config: adapt_config.yaml
```

```bash
hw2 grade submission.yaml --out-dir reports     # the same grader the TAs run
```

`hw2 grade` must run on a fresh clone of your `main` branch; a cell it cannot
evaluate scores 0 with the reason in the report.

## 9. Layout

```
src/hw2_dexhand/
  assets/robots/allegro_hand/   hand and object models (scene_{ball,cube,cylinder,multi}.xml)
  assets/caches/                grasp caches (generated, not shipped)
  tasks/allegro_inhand/         task registration, manager terms, torch quaternion helpers
  conf/ppo/                     Hydra config and task owners (allegro_hold|rotate|reorient|adapt|multi|grasp)
  eval/                         cells, config ownership, policy loading, rollouts, rubric, grading,
                                public_challenges.yaml (the problem 3 public scenarios)
  student/                      your code (empty)
  tools/                        make-caches
  templates/                    student_config.yaml, adapt_config.yaml (the overlay templates)
  cli.py                        the hw2 command
```
