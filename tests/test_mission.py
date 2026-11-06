import threading
import time

from isaac.brain.client import BrainUnavailable, Decision
from isaac.brain.mission import RETRY_START_STEPS, Mission, Phase, Snapshot, build_options


def decision(zone=5, action="go_to_zone", rid="r1"):
    return Decision(action, zone, "why", rid, "m", "0xp", "c", True, 0.1)


class Script:
    """Brain whose answers are given in advance; an exception in the list is raised."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def decide(self, state):
        self.calls.append(state)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class Log:
    def __init__(self):
        self.decisions, self.stops, self.resumes = [], [], 0

    def decision(self, step, trigger, state, d):
        self.decisions.append((trigger, d))

    def safe_stop(self, step, trigger, error):
        self.stops.append((trigger, error))

    def resumed(self, step):
        self.resumes += 1


def snap(step, zone=0, known=None, stuck=False, collected=0):
    return Snapshot(step, 0.0, 0.0, 0.0, zone, stuck, collected, known or [1] * 24, [False] * 24)


def settle(mission, s, tries=200):
    """Call update until the worker's answer lands."""
    for _ in range(tries):
        command = mission.update(s)
        if mission._future is None and mission.phase is not Phase.THINKING:
            return command
        time.sleep(0.005)
    raise AssertionError("worker did not finish")


def make(*answers, **kw):
    brain, log = Script(*answers), Log()
    return Mission(brain, log, "brief", kw.pop("limit", 10_000), **kw), brain, log


def test_wheels_stay_stopped_until_0g_answers():
    mission, _, log = make(decision())
    assert mission.update(snap(0)).drive is False
    command = settle(mission, snap(1))
    assert command.drive is True and command.focus_zone == 5
    assert log.decisions[0][0] == "mission_start"


def test_0g_failure_stops_the_rover_and_invents_nothing():
    mission, _, log = make(BrainUnavailable("down"))
    command = settle(mission, snap(1))
    assert command.drive is False and command.focus_zone == -1
    assert command.phase is Phase.SAFE_STOP
    assert log.stops == [("mission_start", "down")] and log.decisions == []


def test_rover_never_moves_during_an_outage():
    mission, brain, _ = make(*[BrainUnavailable("down")] * 50)
    step = 0
    for _ in range(4000):
        step += 1
        assert mission.update(snap(step)).drive is False
        time.sleep(0)
    assert len(brain.calls) >= 2


def test_retries_back_off_then_recover():
    mission, brain, log = make(BrainUnavailable("down"), decision(zone=9))
    settle(mission, snap(1))
    assert mission.update(snap(RETRY_START_STEPS - 1)).drive is False
    assert len(brain.calls) == 1
    mission.update(snap(1 + RETRY_START_STEPS))
    time.sleep(0.05)
    command = mission.update(snap(2 + RETRY_START_STEPS))
    assert command.drive is True and command.focus_zone == 9 and log.resumes == 1


def test_failure_at_a_later_checkpoint_also_stops():
    mission, _, log = make(decision(zone=5), BrainUnavailable("down"), zone_budget_steps=10)
    settle(mission, snap(1))
    mission.update(snap(20))
    command = settle(mission, snap(21))
    assert command.drive is False and log.stops[0][0] == "zone_budget_spent"


def test_finish_ends_the_mission():
    mission, _, _ = make(decision(zone=None, action="finish"))
    assert settle(mission, snap(1)).phase is Phase.DONE


def test_time_limit_ends_the_mission_without_a_decision():
    mission, brain, _ = make(decision(), limit=100)
    settle(mission, snap(1))
    command = mission.update(snap(100))
    assert command.phase is Phase.DONE and command.drive is False and len(brain.calls) == 1


def test_cleared_zone_triggers_a_checkpoint_after_settling():
    mission, brain, log = make(decision(zone=5), decision(zone=6), settle_steps=3)
    settle(mission, snap(1))
    empty = [0] * 24
    assert mission.update(snap(2, zone=5, known=empty, collected=2)).drive is True
    assert mission.update(snap(3, zone=5, known=empty, collected=2)).drive is True
    settle(mission, snap(4, zone=5, known=empty, collected=2))
    assert log.decisions[1][0] == "zone_done"
    assert brain.calls[1]["previous_assignments"][0]["balls_collected"] == 2


def test_stuck_triggers_a_checkpoint():
    mission, _, log = make(decision(zone=5), decision(zone=6), stuck_steps=2)
    settle(mission, snap(1))
    mission.update(snap(2, stuck=True))
    mission.update(snap(3, stuck=True))
    settle(mission, snap(4))
    assert log.decisions[1][0] == "stuck"


def test_state_sent_to_0g_has_only_what_the_rover_knows():
    mission, brain, _ = make(decision())
    known = [0] * 24
    known[3] = 2
    settle(mission, snap(1, known=known))
    assert brain.calls[0]["known_balls_by_zone"] == {"3": 2}


def test_options_list_balls_first_and_mark_the_net():
    from isaac.brain.mission import build_options
    known = [0] * 24
    known[7] = 2
    known[3] = 5
    s = Snapshot(0, -12.0, -5.0, 0.0, 0, False, 0, known, [True] * 24)
    options = build_options(s, set())
    assert [o["zone"] for o in options] == [3, 7]
    assert options[0]["crosses_net"] is True and options[1]["crosses_net"] is False


def test_zones_that_gave_nothing_are_not_offered_again():
    from isaac.brain.mission import build_options
    known = [0] * 24
    known[9] = 2
    s = Snapshot(0, 0.0, 0.0, 0.0, 0, False, 0, known, [True] * 24)
    assert build_options(s, {9}) == []


def test_mission_ends_without_calling_0g_when_nothing_is_left():
    mission, brain, _ = make(decision())
    s = Snapshot(1, 0.0, 0.0, 0.0, 0, False, 0, [0] * 24, [True] * 24)
    command = mission.update(s)
    assert command.phase is Phase.DONE and command.drive is False
    assert mission.end_reason == "no_zones_left" and brain.calls == []


def test_option_list_is_capped():
    from isaac.brain.mission import MAX_OPTIONS, build_options
    assert len(build_options(snap(0, known=[0] * 24), set())) == MAX_OPTIONS


def test_only_one_unexplored_zone_is_offered_once_balls_are_known():
    from isaac.brain.mission import build_options
    known = [0] * 24
    known[7] = 2
    options = build_options(snap(0, known=known), set())
    assert [o["balls"] for o in options] == [2, 0]


class Gate:
    """Brain that holds its answer until released, like a slow network."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []
        self.release = threading.Event()

    def decide(self, state):
        self.calls.append(state)
        self.release.wait(2)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def driving_mission(brain, log=None, **kwargs):
    """A mission already driving to zone 5, started by a real answer."""
    gate_open = getattr(brain, "release", None)
    if gate_open:
        gate_open.set()
    m = Mission(brain, log or Log(), "brief", 10_000, settle_steps=3, **kwargs)
    settle(m, snap(0))
    if gate_open:
        gate_open.clear()
    return m


def wait_for(condition):
    for _ in range(200):
        if condition():
            return
        time.sleep(0.005)
    raise AssertionError("condition never met")


def test_next_zone_is_asked_for_while_the_rover_keeps_driving():
    brain = Gate(decision(5), decision(7))
    m = driving_mission(brain)
    empty = [0] * 24
    empty[7] = 2
    command = m.update(snap(10, zone=5, known=empty))
    wait_for(lambda: len(brain.calls) == 2)
    assert command.drive and command.focus_zone == 5
    assert brain.calls[1]["trigger"] == "zone_done"
    assert 5 in [h["zone"] for h in brain.calls[1]["previous_assignments"]]


def test_answer_ready_at_checkpoint_means_no_stop():
    brain = Gate(decision(5), decision(7))
    log = Log()
    m = driving_mission(brain, log)
    empty = [0] * 24
    empty[7] = 2
    m.update(snap(10, zone=5, known=empty))
    brain.release.set()
    wait_for(lambda: m._future is not None and m._future.done())
    assert m.update(snap(11, zone=5, known=empty)).drive
    command = m.update(snap(12, zone=5, known=empty))
    assert command.drive and command.focus_zone == 7
    assert m.phase is Phase.DRIVING and len(log.decisions) == 2


def test_late_answer_stops_the_wheels_until_it_lands():
    brain = Gate(decision(5), decision(7))
    m = driving_mission(brain)
    empty = [0] * 24
    empty[7] = 2
    for step in (10, 11):
        m.update(snap(step, zone=5, known=empty))
    command = m.update(snap(12, zone=5, known=empty))
    assert not command.drive and m.phase is Phase.THINKING
    brain.release.set()
    wait_for(lambda: m._future.done())
    assert m.update(snap(13, zone=5, known=empty)).focus_zone == 7


def test_failed_early_ask_stops_at_once_and_records_no_decision():
    brain = Gate(decision(5), BrainUnavailable("down"))
    log = Log()
    m = driving_mission(brain, log)
    empty = [0] * 24
    empty[7] = 2
    m.update(snap(10, zone=5, known=empty))
    brain.release.set()
    wait_for(lambda: m._future.done())
    command = m.update(snap(11, zone=5, known=empty))
    assert not command.drive and command.focus_zone == -1
    assert m.phase is Phase.SAFE_STOP
    assert len(log.decisions) == 1 and len(log.stops) == 1


def test_answer_is_dropped_if_new_balls_appear_in_the_zone():
    brain = Gate(decision(5), decision(7), decision(5))
    log = Log()
    m = driving_mission(brain, log)
    empty = [0] * 24
    empty[7] = 2
    m.update(snap(10, zone=5, known=empty))
    wait_for(lambda: len(brain.calls) == 2)
    busy = list(empty)
    busy[5] = 1
    m.update(snap(11, zone=5, known=busy))
    brain.release.set()
    wait_for(lambda: m._future is None or m._future.done())
    command = m.update(snap(12, zone=5, known=busy))
    assert command.drive and command.focus_zone == 5
    assert m._future is None and len(log.decisions) == 1


def test_budget_ask_happens_before_the_budget_runs_out():
    brain = Gate(decision(5), decision(7))
    m = driving_mission(brain, zone_budget_steps=100)
    m.update(snap(96))
    assert len(brain.calls) == 1
    m.update(snap(97))
    wait_for(lambda: len(brain.calls) == 2)
    assert brain.calls[1]["trigger"] == "zone_budget_spent"


def test_far_side_is_held_back_while_near_side_has_work():
    known = [0] * 24
    known[1] = 1   # near side (left of the net)
    known[4] = 5   # far side, more balls
    s = Snapshot(0, -5.0, 0.0, 0.0, 1, False, 0, known, [True] * 24)
    assert [o["zone"] for o in build_options(s, set())] == [4, 1]
    assert [o["zone"] for o in build_options(s, set(), near_left=True)] == [1]


def test_far_side_opens_when_near_side_is_done():
    known = [0] * 24
    known[4] = 5
    s = Snapshot(0, -5.0, 0.0, 0.0, 1, False, 0, known, [True] * 24)
    assert [o["zone"] for o in build_options(s, set(), near_left=True)] == [4]


def test_near_side_is_where_the_mission_started():
    brain = Script(decision(1))
    known = [0] * 24
    known[1], known[4] = 1, 5
    m = Mission(brain, Log(), "brief", 10_000, near_side_first=True)
    settle(m, Snapshot(0, 8.0, 0.0, 0.0, 16, False, 0, known, [True] * 24))
    assert [o["zone"] for o in brain.calls[0]["options"]] == [4]
