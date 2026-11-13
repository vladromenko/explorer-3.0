"""Observation availability and measured command timing are different facts."""


def action_available_ns(row: dict) -> int:
    value = row.get('action_available_ns', row.get('action_send_end_ns'))
    if value is None:
        raise ValueError('Command availability time is required')
    return int(value)


def validate_timing(row: dict) -> None:
    observation = row['observation_timestamp_ns']
    available = action_available_ns(row)
    start, end = row.get('action_send_start_ns'), row.get('action_send_end_ns')
    if available < observation:
        raise ValueError('Command target was not available after the observation')
    if start is None or end is None:
        if start is not None or end is not None:
            raise ValueError('Incomplete measured command interval')
        if row.get('action_time_kind') != 'command_in_force_at_next_agent_snapshot':
            raise ValueError('Missing command interval without an explicit snapshot contract')
    elif not observation <= start <= end <= available:
        raise ValueError('Noncausal measured command interval')
    effective = row.get('action_effective_ns')
    if effective is not None and effective > available:
        raise ValueError('Command effective time is in the future')


def send_latency_s(row: dict) -> float | None:
    # Includes historical records: an equal pair of snapshot timestamps is not a measured zero latency.
    if row.get('action_time_kind') == 'command_in_force_at_next_agent_snapshot':
        return None
    start, end = row.get('action_send_start_ns'), row.get('action_send_end_ns')
    return (end-start)/1e9 if start is not None and end is not None else None
