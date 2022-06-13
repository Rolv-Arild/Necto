
![Necto Half-Flip In Action](https://github.com/Rolv-Arild/Necto/blob/master/nectoGifs/nectoHalfFlip.gif)


# What is this?

This is Tecko, our community machine learning Rocket League bot. 
This is the 3rd generation of the bot and its learning to play 1's, 2's, and 3's thanks to RLGym.
Our end goalis making a version that can take down pros!

Fun fact, Tecko has the same number of neurons as a real life gecko!

V1: Necto <br/>
V2: Nexto <br/>
V3: Tecko


[Watch Nexto V2 go up against RLCS Pros FairyPeak, Breezi, Aztral, Kaydop, and more on our YouTube channel!](https://www.youtube.com/c/RLGym/videos)

[SunlessKhan](https://www.youtube.com/watch?v=owhz5RSX0go) and [Rocket Sledge](https://www.youtube.com/watch?v=LO4h8djNB50&t=387s)
have both made videos about Nexto!

[Watch Necto V1 win the 2022 RLBot Championship!](https://youtu.be/XVIxZA6gFRI?t=13753)

# How does it work?

These bots are trained with [Deep Reinforcement Learning](https://wiki.pathmind.com/deep-reinforcement-learning), 
a type of Machine Learning. We have several games playing at super speeds behind the scenes while the data is collected and learned from.
We ingest these games using a custom built distributed learning system.

We define rewards that the bot tries to achieve. Over time, behavior that leads to more reward gets reinforced, which leads to 
better Rocket League play.

# Can I play against it? 

Yup! Go download the [RLBot pack](https://rlbot.org/) and Nexto and Necto are some of the bots that you can play against.
 Make sure fps is set to 120 and VSync is turned off.

# What Rocket League rank is the bot?

Tecko is still being trained so we don't yet know.

Nexto is low-mid Grand Champion, depending on the version

Necto is around high Diamond. 




# Can I watch it learn?

Yes! Check out our [Twitch stream here](https://www.twitch.tv/rlgym).

[Graphs are also available](https://wandb.ai/rolv-arild/rocket-learn) for our fellow nerds.


# Could it learn by looking at Pro/SSL replays?

It can't directly learn from replays. However, we like to start matches from moments found in high level
replays. This gives the bot more scenarios to see and learn from.


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



