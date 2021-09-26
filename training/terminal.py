from rlgym.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition


def NectoTerminalCondition(tick_skip=8):
    return (
        NoTouchTimeoutCondition(round(30 * 120 / tick_skip)),
        GoalScoredCondition()
    )
