from rlgym.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition, TimeoutCondition


def NectoTerminalCondition(tick_skip=8):
    return (
        NoTouchTimeoutCondition(round(30 * 120 / tick_skip)),
        GoalScoredCondition()
    )


def NectoHumanTerminalCondition(tick_skip=8):
    return (
        TimeoutCondition(round(30 * 120 / tick_skip)),
        GoalScoredCondition()
    )
