
![Necto Half-Flip In Action](https://github.com/Rolv-Arild/Necto/blob/master/nectoGifs/nectoHalfFlip.gif)


# What is this?

This is Necto, the RLGym community bot. It's a bot that's learning to play 1's, 2's, and 3's thanks to RLGym.
Rocket League is a hard game though so we're all working together to give it the experience it needs.

[Watch Nexto go up against Extra of Team BDS!](https://www.twitch.tv/videos/1443124771)

[Watch Nexto go up against Polar of Team Singularity!](https://www.twitch.tv/videos/1440172971)

[Watch Necto win the 2022 RLBot Championship!](https://youtu.be/XVIxZA6gFRI?t=13753)

# How does it work?

Necto (and the new version Nexto) are trained with [Deep Reinforcement Learning](https://wiki.pathmind.com/deep-reinforcement-learning), 
a type of Machine Learning. We have several games playing at super speeds behind the scenes while the data is collected and learned from.

We define rewards that Necto tries to achieve. Over time, behavior that leads to more reward gets reinforced, which leads to 
better Rocket League play.

# Can I play against it? 

Yup! Go download the [RLBot pack](https://rlbot.org/) and Necto is one of several bots that you can play against.
 Make sure fps is set to 120 and VSync is turned off.

# What Rocket League rank is Necto?

We estimate the RLBot version of Necto is high Diamond. 

Nexto is still in training but it appears to be low Grand Champion at the moment.


# Can I watch it learn?

Yes! Check out our [Twitch stream here](https://www.twitch.tv/rlgym).

[Graphs are also available](https://wandb.ai/rolv-arild/rocket-learn) for our fellow nerds.


# Could it learn by looking at Pro/SSL replays?

It can't directly learn from replays. However, we like to start matches from moments found in high level
replays. This gives Necto more scenarios to see and learn from.


# Could it learn by playing against me?

We're working on it!


# Can donate my compute to help it learn faster?

If you're interested in offering your compute, first check our discord to see if we need more helpers. Then, make sure you have downloaded:
- Rocket League (Epic version is required)
- bakkesmod
- python >=3.8
- git

## To start computing

1. clone repo
2. start cmd as admin
3. create symlink to repo\training\training repo\training (mklink /D path\to\repo\training\training path\to\repo\training)
4. close cmd
6. open repo\training in windows explorer or non-admin cmd
7. Run runDistributedWorker.bat
8. enter your name + server ip + server password

when asked for the IP and password, dm Soren and he'll fill you in. 



