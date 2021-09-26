# RLGymExampleBot
RLGym example bot for the RLBot framework, based on the official RLBotPythonExample

## How to use this 

This bot runs the Actor class in `src/actor.py`, you're expected to replace that with the output of your model

By default we use DefaultObs from RLGym, AdvancedObs is also available in this project. 

You can also provide your own custom ObservationBuilder by copying it over and replacing the `rlgym` imports with `rlgym_compat` (check `src/obs/` for some examples)

## Changing the bot

- Bot behavior is controlled by `src/bot.py`
- Bot appearance is controlled by `src/appearance.cfg`

See https://github.com/RLBot/RLBotPythonExample/wiki for documentation and tutorials.

## Running a match

You can start a match by running `run.py`, the match config for it is in `rlbot.cfg`
