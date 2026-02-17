# Improving Policy Exploitation in Online Reinforcement Learning with Instant Retrospect Action
=======
# GPU(+CPU) version
$ conda install -c pytorch -c nvidia faiss-gpu=1.8.0

# Train  IRA
$ python main.py --env=HalfCheetah-v3   --device=cuda:0  --seed= --beta=1.0 --k=10 --warmup_timestamps=4000   --action_buffer_size=200000 --policy_freq=1 --start_timesteps=10000 --policy=IRA

