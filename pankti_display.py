"""Historical pairing helpers for baseline replays; production shows one line."""
from bisect import bisect_left, bisect_right


def identity(line):
    return line.get('pankti_id'), line.get('bani_id'), line.get('bani_position')


def consecutive(first, second, next_by_id):
    if not first.get('pankti_id') or not second.get('pankti_id'):
        return False
    if first.get('bani_id') is not None or second.get('bani_id') is not None:
        ordered = (first.get('bani_id') is not None
                   and first.get('bani_id') == second.get('bani_id')
                   and isinstance(first.get('bani_position'), int)
                   and second.get('bani_position') == first['bani_position']+1)
        if first.get('shabad_id') == second.get('shabad_id'):
            ordered = ordered and next_by_id.get(first['pankti_id']) == second['pankti_id']
        return ordered
    return (first.get('shabad_id') == second.get('shabad_id')
            and next_by_id.get(first['pankti_id']) == second['pankti_id'])


def display_lines(timeline, playhead, selected, next_by_id, max_seconds=6.0):
    """Look ahead within the delayed audio, using confirmed matches only.

    Keep two fast consecutive lines in reading order while either is heard.
    A backward jump, a skipped canonical line, a greeting or an unrelated
    shabad breaks the pair. Auto-next alone cannot establish a sung pair.
    """
    if not selected or selected.get('waiting_for_shabad'):
        return []
    single = [dict(selected)]
    left = bisect_left(timeline, playhead-2*max_seconds, key=lambda item: item[0])
    if left:
        left -= 1
        while left and timeline[left][1].get('match_mode') == 'bani-auto-next':
            left -= 1
    right = bisect_right(timeline, playhead+max_seconds, key=lambda item: item[0])
    heard = []
    for at, line in timeline[left:right]:
        if line.get('match_mode') == 'bani-auto-next':
            continue
        if not heard or identity(heard[-1][1]) != identity(line):
            heard.append((at, dict(line)))
        elif line.get('last_word_timestamp') is not None:
            heard[-1][1]['last_word_timestamp'] = max(
                heard[-1][1].get('last_word_timestamp', heard[-1][0]), line['last_word_timestamp'])
    position = next((i for i in range(len(heard)-1, -1, -1)
                     if heard[i][0] <= playhead), None)
    if position is None:
        return single
    for start in (position, position-1):
        if start < 0 or start+1 >= len(heard):
            continue
        (first_at, first), (second_at, second) = heard[start:start+2]
        last_word = min(second_at, first.get('last_word_timestamp', first_at))
        if (second_at > first_at and 0 <= second_at-last_word <= max_seconds
                and max(first_at, second_at-max_seconds) <= playhead < second_at+max_seconds
                and consecutive(first, second, next_by_id)
                and identity(selected) in (identity(first), identity(second))):
            return [dict(first, audio_timestamp=first_at),
                    dict(second, audio_timestamp=second_at)]
    return single
