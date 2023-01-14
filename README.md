
![Nexto Teamplay](https://github.com/Rolv-Arild/Necto/blob/master/nectoGifs/nexto-clip.gif)


# What is this?

This is Necto, our community machine learning Rocket League bot. 
It has learned to play 1's, 2's, and 3's thanks to [RLGym](https://github.com/lucas-emery/rocket-league-gym).
Our end goal is making a version that can take down pros!

So far, we've made 3 versions: </br>
V1: Necto - Around Diamond level. </br>
V2: Nexto - Approximately Grand Champion 1 level in 1v1, 2v2 and 3v3 (top 0.12%, 0.95%, 0.46% of the playerbase respectively)  </br>
V3: Tecko - Canceled due to lack of improvement.  </br>
V4: N/A - Moved to new project. </br>

Some videos:
- [2v1 showmatches against pros on our YouTube channel](https://www.youtube.com/c/RLGym/videos).
- [SunlessKhan going up against Nexto](https://www.youtube.com/watch?v=owhz5RSX0go)
- [Rocket Sledge pitting Grand Champions against Nexto](https://www.youtube.com/watch?v=LO4h8djNB50&)
- [Rocket Science's video about how Nexto works and some exploits](https://www.youtube.com/watch?v=jQHt2O0PkCQ&t=518s)
- [Watch Necto V1 win the 2022 RLBot Championship](https://youtu.be/XVIxZA6gFRI)

# How does it work?

These bots are trained with [Deep Reinforcement Learning](https://wiki.pathmind.com/deep-reinforcement-learning), 
a type of Machine Learning. We have several games playing at super speeds behind the scenes while the data is collected and learned from.
We ingest these games using a custom built distributed learning system.

We define rewards that the bot tries to achieve. Over time, behavior that leads to more reward gets reinforced, which leads to 
better Rocket League play.

# Can I play against it? 

Yup! Go download the [RLBot pack](https://rlbot.org/) and Nexto and Necto are some of the bots that you can play against.
 Make sure fps is set to 120 and VSync is turned off.


# Can I watch it learn?

We are currently not training it, but Necto/Nexto/Tecko were shown on our [Twitch stream](https://www.twitch.tv/rlgym), which is currently streaming other bots training.

[Graphs are also available](https://wandb.ai/rolv-arild/necto) for our fellow nerds.


# Could it learn by looking at Pro/SSL replays?

Yes! Inspired by [Video PreTraining](https://arxiv.org/abs/2206.11795), we can now use replay files to learn from human gameplay and see years of gameplay before
setting a wheel on the field. Occasionally, we showcase its progress on our Twitch stream. In the future, we plan to follow up this "jumpstarted" training with live reinforcement learning.

[Here's a repository containing code and explanation](https://www.youtube.com/watch?v=oz5yZc9ULAc)


# Could it learn by playing against me?

In theory it could, however in practice the rate at which we can collect data by pitting them against each other at high speed would be very hard to match by using humans (we'd need hundreds of people playing at once). The humans would ideally also be approximately equal skill to the bot at all points.


# Can I donate my compute to help it learn faster?

We're currently not accepting compute donations but thanks for your interest!


# What is Nexto+?

Nexto+ is a secret post-training upgrade to Nexto that increases its already impressive skill. It is not available for play but may make appearances in future RLBot tournaments.


# What is Toxic Nexto?

Toxix Nexto is a version of Nexto at the same skill level but provides that authentic Rocket League experience of harsh words and bad vibes. Its equally mean to its opponents as to its teammates.

It can be played against in the RLBot pack.



