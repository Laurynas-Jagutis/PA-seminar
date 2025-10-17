import copy
import random
import numpy as np
import datetime
from pathlib import Path
from pm4py.objects.process_tree import semantics
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

from conceptdrift.source.evolution import evolve_tree_randomly_gs
from conceptdrift.source.event_log_controller import combine_two_logs, add_duration_to_log, get_timestamp_log
from conceptdrift.source.process_tree_controller import generate_specific_trees


# SOme smaller recurring helper functions to reduce the size of the event log creation functions
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def add_timing_and_metadata(event_log, num_traces, drift_points, drift_type, added_acs, deleted_acs, moved_acs):
    date = datetime.datetime.strptime('20/8/3 8:0:0', '%y/%d/%m %H:%M:%S')
    add_duration_to_log(event_log, date, 1, 14000)
    drift_times = [get_timestamp_log(event_log, num_traces, dp / num_traces) for dp in drift_points]
    info = (
        f"drift perspective: control-flow; drift type: {drift_type}; "
        f"drift points: {drift_points}; drift timestamps: {drift_times}; "
        f"activities added: {added_acs}; activities deleted: {deleted_acs}; activities moved: {moved_acs}"
    )
    event_log.attributes['drift info'] = info
    return event_log

def generate_base_tree():
    """Generate the starting process tree."""
    trees = generate_specific_trees('middle')
    return trees[0]

# Creating event logs with sudden concept drift
def generate_sudden_drift_log(num_traces=1500, drift_points=(500, 1000), change_proportion=0.2, random_seed=None):
    if random_seed is not None:
        set_random_seed(random_seed)

    ver_current = generate_base_tree()
    added_acs_total, deleted_acs_total, moved_acs_total = [], [], []

    logs = []
    prev_index = 0

    for point in list(drift_points) + [num_traces]:
        seg_length = point - prev_index
        log_segment = semantics.generate_log(ver_current, seg_length)
        logs.append(log_segment)
        if point != num_traces:
            ver_current, deleted, added, moved = evolve_tree_randomly_gs(copy.deepcopy(ver_current), change_proportion)
            added_acs_total += added
            deleted_acs_total += deleted
            moved_acs_total += moved
        prev_index = point

    event_log = logs[0]
    for l in logs[1:]:
        event_log = combine_two_logs(event_log, l)

    event_log = add_timing_and_metadata(event_log, num_traces, drift_points, "sudden", added_acs_total, deleted_acs_total, moved_acs_total)
    return event_log


# Creating event logs with gradual concept drift
def generate_gradual_drift_log(num_traces=1500, drift_points=(500, 1000), transition_window=100, change_proportion=0.2, random_seed=None):
    if random_seed is not None:
        set_random_seed(random_seed)

    ver_current = generate_base_tree()
    added_acs_total, deleted_acs_total, moved_acs_total = [], [], []

    logs = []
    prev_index = 0

    for point in list(drift_points) + [num_traces]:
        transition_start = max(prev_index, point - transition_window)

        # Stable region
        if transition_start > prev_index:
            log_stable = semantics.generate_log(ver_current, transition_start - prev_index)
            logs.append(log_stable)

        # Transition region
        if point != num_traces:
            new_model, deleted, added, moved = evolve_tree_randomly_gs(copy.deepcopy(ver_current), change_proportion)
            added_acs_total += added
            deleted_acs_total += deleted
            moved_acs_total += moved
            mix_length = point - transition_start
            mixed_logs = []
            for i in range(mix_length):
                p_new = (i + 1) / mix_length
                chosen_model = new_model if random.random() < p_new else ver_current
                log_part = semantics.generate_log(chosen_model, 1)
                mixed_logs.append(log_part)
            # Combine both parts
            combined_mix = mixed_logs[0]
            for ml in mixed_logs[1:]:
                combined_mix = combine_two_logs(combined_mix, ml)
            logs.append(combined_mix)
            ver_current = new_model
        prev_index = point

    event_log = logs[0]
    for l in logs[1:]:
        event_log = combine_two_logs(event_log, l)

    event_log = add_timing_and_metadata(event_log, num_traces, drift_points, "gradual", added_acs_total, deleted_acs_total, moved_acs_total)
    return event_log


# Generating recurring concept drift event logs
def generate_recurring_drift_log(num_traces=1500, drift_points=(500, 1000), change_proportion=0.2, random_seed=None):
    if random_seed is not None:
        set_random_seed(random_seed)

    base_model = generate_base_tree()
    evolved_model, *_ = evolve_tree_randomly_gs(copy.deepcopy(base_model), change_proportion)

    models = [base_model, evolved_model, base_model]
    logs = []
    prev_index = 0

    for i, point in enumerate(list(drift_points) + [num_traces]):
        seg_length = point - prev_index
        model = models[i % len(models)]
        log_segment = semantics.generate_log(model, seg_length)
        logs.append(log_segment)
        prev_index = point

    event_log = logs[0]
    for l in logs[1:]:
        event_log = combine_two_logs(event_log, l)

    event_log = add_timing_and_metadata(event_log, num_traces, drift_points, "recurring", [], [], [])
    return event_log


def generate_all_drifts(output_dir="AdditionalLogs", seeds=range(10)):
    Path(output_dir).mkdir(exist_ok=True)

    for seed in seeds:
        # Generate sudden drift event logs
        sudden_log = generate_sudden_drift_log(num_traces=1500, drift_points=(500, 1000), random_seed=seed)
        xes_exporter.apply(sudden_log, f"{output_dir}/sudden_drift_{seed:02d}.xes")

        # Generate gradual drift event logs
        gradual_log = generate_gradual_drift_log(num_traces=1500, drift_points=(500, 1000), random_seed=seed)
        xes_exporter.apply(gradual_log, f"{output_dir}/gradual_drift_{seed:02d}.xes")

        # Generate recurring drift event logs
        recurring_log = generate_recurring_drift_log(num_traces=1500, drift_points=(500, 1000), random_seed=seed)
        xes_exporter.apply(recurring_log, f"{output_dir}/recurring_drift_{seed:02d}.xes")

        print(f"Generated all drifts for seed {seed}")


if __name__ == "__main__":
    generate_all_drifts()
    print("Drift logs generated successfully.")
