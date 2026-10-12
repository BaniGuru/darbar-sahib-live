"""Ordered bani following, separate from the existing shabad search.

Positions identify occurrences, not just line IDs: a bani can reuse a pankti
in several groups. Variant evidence comes only from heard lines; predictions
never confirm themselves or discard the alternative Aarti/Rehras route.
"""
from collections import defaultdict
import re


def bani_family(name):
    name = re.sub(r"[^a-z]", "", name.casefold())
    if name.startswith("aarti"):
        return "aarti"
    if name.startswith(("rehras", "rehraas")):
        return "rehras"
    if name.startswith("asa") and ("var" in name or "vaar" in name):
        return "asa-ki-vaar"
    return None


class OrderedBaniMatcher:
    def __init__(self, routes, similarity, normalise):
        self.routes = routes
        self.similarity = similarity
        self.normalise = normalise
        self.members = defaultdict(list)
        self.family_routes = defaultdict(list)
        for bani_id, route in routes.items():
            self.family_routes[route["family"]].append(bani_id)
            for position, row in enumerate(route["lines"]):
                self.members[row["pankti_id"]].append((bani_id, position))
        self.family = None
        self.positions = {}
        self.evidence = defaultdict(int)
        self.selected = None
        self.predicted = False
        self.end_time = None
        self.end_tail = ()
        self.last_observed = None
        self.last_bani_audio = None
        self.finished = False

    @staticmethod
    def spoken_tokens(row):
        return [word for word in row["_tokens"]
                if not word.isnumeric() and word not in {"ਰਹਾਉ", "ਰਹਾਓ"}]

    def _position(self, bani_id, pankti_id):
        indexes = [p for b, p in self.members.get(pankti_id, ()) if b == bani_id]
        current = self.positions.get(bani_id, -1)
        return min(indexes, key=lambda p: (abs(p - current), p < current), default=None)

    def _choose(self, eligible):
        return max(eligible, key=lambda b: (
            self.evidence[b], b == self.selected, -len(self.routes[b]["lines"])))

    def _payload(self, bani_id, position, score, query_end, *, auto=False):
        route = self.routes[bani_id]
        row = route["lines"][position]
        return {key: row.get(key) for key in (
            "pankti_id", "shabad_id", "order_id", "punjabi", "legacy_gurmukhi", "english"
        )} | {"score": round(score, 4), "bani_id": bani_id,
             "bani_name": route["name"], "line_group": row["line_group"],
             "bani_position": position, "match_mode": "bani-auto-next" if auto else "bani",
             "_query_end": query_end, "_bani_direct": True,
             "_bani_auto_next": auto,
             "_bani_observed_shabad_id": route["lines"][self.positions[bani_id]]["shabad_id"]}

    def _heard(self, transcript, message):
        tokens = self.normalise(transcript).split()[-12:]
        records = []
        for key in ("committed_words", "partial_words"):
            for word in (message.get(key) or [])[-24:]:
                try:
                    records.append((float(word["start"]), float(word["end"]),
                                    self.normalise(word["text"]).split()))
                except (KeyError, ValueError, TypeError):
                    continue
        if self.end_time is not None and records:
            fresh = [token for start, _, words in records if start > self.end_time + .001
                     for token in words]
        elif self.end_tail:
            overlap = next((n for n in range(min(len(tokens), len(self.end_tail)), 0, -1)
                            if tuple(tokens[:n]) == self.end_tail[-n:]), 0)
            fresh = tokens[overlap:]
        else:
            fresh = tokens
        return tokens, fresh[-12:], records

    def _end_reached(self, row, tokens):
        expected = self.spoken_tokens(row)
        size = min(3, len(expected))
        if size < 2 or len(tokens) < size:
            return None
        # One update can include the line ending plus the first word(s) of
        # what follows. Anchor at that ending, not at the newest word.
        for end in range(len(tokens), max(size - 1, len(tokens) - 5), -1):
            scores = [self.similarity(a, b) for a, b in zip(expected[-size:], tokens[end-size:end])]
            if min(scores) >= .60 and sum(scores) / size >= .88:
                return end
            # Once this line is identified, a very strong terminal bigram
            # also confirms its end when ASR damaged the preceding word.
            if size == 3 and min(scores[-2:]) >= .88 and sum(scores[-2:]) / 2 >= .93:
                return end
        return None

    def _complete(self, transcript, message, score=1.0):
        tokens, _, records = self._heard(transcript, message)
        position = self.positions[self.selected]
        row = self.routes[self.selected]["lines"][position]
        result = self._payload(self.selected, position, score, len(tokens))
        ending = self._end_reached(row, tokens)
        if ending is None:
            return result
        flattened = [(start, token) for start, _, words in records for token in words]
        if len(flattened) >= len(tokens) and [x[1] for x in flattened[-len(tokens):]] == tokens:
            ending_time = flattened[len(flattened) - len(tokens) + ending - 1][0]
        else:
            ending_time = next((start for start, token in reversed(flattened)
                                if token == tokens[ending-1]), None)
        if ending_time is not None and self.end_time is not None and ending_time <= self.end_time:
            return result  # The old ending cannot finish a newly sung repeat.
        self.end_time = ending_time
        self.end_tail = tuple(tokens[:ending])
        self.predicted = True
        eligible = [b for b, p in self.positions.items()
                    if p + 1 < len(self.routes[b]["lines"])
                    and self.routes[b]["lines"][p]["pankti_id"] == row["pankti_id"]]
        if not eligible:
            self.family = None
            self.finished = True
            return result
        if self.selected not in eligible:
            # The best-supported version has ended. Keep its last pankti
            # rather than inventing a continuation from a less-supported
            # variant; actual additional words can still select that route.
            return result
        return self._payload(self.selected, self.positions[self.selected] + 1,
                             score, ending, auto=True)

    def adopt(self, match, transcript, message):
        """Observe an accepted ordinary match, or resume after an interruption."""
        if self.predicted and self.selected is not None:
            p = self.positions[self.selected]
            row = self.routes[self.selected]["lines"][p]
            if str(match["pankti_id"]) == row["pankti_id"] and p+1 < len(self.routes[self.selected]["lines"]):
                tokens, fresh, _ = self._heard(transcript, message)
                if not fresh or int(match.get("_query_end") or len(tokens)) <= len(tokens)-len(fresh):
                    return self._payload(self.selected, p+1, float(match.get("score",1)),
                                         int(match.get("_query_end") or len(tokens)), auto=True)
        memberships = self.members.get(str(match["pankti_id"]), ())
        if not memberships:
            self.family = None
            return match
        if (self.family is None and all(self.routes[b]["lines"][p].get("_line_type") == 1
                                        for b, p in memberships)):
            # A shared invocation does not identify a bani or its version.
            return match
        if self.finished:
            self.positions = {}
            self.evidence.clear()
            self.selected = None
            self.last_observed = None
            self.end_time = None
            self.end_tail = ()
            self.last_bani_audio = None
            self.finished = False
        families = {self.routes[b]["family"] for b, _ in memberships}
        previous_family = self.routes[self.selected]["family"] if self.selected else None
        # A shabad inserted into Asa Ki Vaar may itself occur in Aarti or
        # Rehras. Membership alone must not turn that interruption into a
        # different bani session. Retain the Asa resume context for the same
        # five-minute hold used by the ordinary live matcher.
        audio_time = float(message.get("window_end") or 0)
        if (previous_family == "asa-ki-vaar" and "asa-ki-vaar" not in families
                and self.last_bani_audio is not None
                and audio_time - self.last_bani_audio <= 300):
            self.family = None
            return match
        family = (self.family if self.family in families else
                  previous_family if previous_family in families else
                  self.routes[memberships[0][0]]["family"])
        if self.family != family and not any(self.routes[b]["family"] == family
                                             for b in self.positions):
            self.positions = {}
            self.evidence.clear()
            self.selected = None
            self.end_time = None
            self.end_tail = ()
            self.last_observed = None
        self.family = family
        self.last_bani_audio = audio_time
        eligible = []
        for bani_id in self.family_routes[family]:
            p = (match["bani_position"] if match.get("bani_id") == bani_id
                 else self._position(bani_id, str(match["pankti_id"])))
            if p is not None:
                self.positions[bani_id] = p
                eligible.append(bani_id)
        marker = (str(match["pankti_id"]), tuple((b, self.positions[b]) for b in eligible))
        if marker != self.last_observed:
            for bani_id in eligible:
                self.evidence[bani_id] += 1
            self.last_observed = marker
        self.selected = self._choose(eligible)
        self.predicted = False
        return self._complete(transcript, message, float(match.get("score", 1)))

    def match(self, transcript, message):
        if self.family is None:
            return None
        tokens, fresh, _ = self._heard(transcript, message)
        if self.predicted and not fresh:
            # A revised copy of the just-finished word cannot advance again.
            return None
        heard = fresh if self.end_time is not None or self.end_tail else tokens
        candidates = []
        for bani_id in self.family_routes[self.family]:
            if bani_id not in self.positions:
                continue
            current = self.positions[bani_id]
            rows = self.routes[bani_id]["lines"]
            # Nearby rows let optional invocations be skipped, repeats be
            # restored, and either variant win as its actual text is heard.
            for p in range(max(0, current - 2), min(len(rows), current + 7)):
                expected = self.spoken_tokens(rows[p])
                if not expected or not heard:
                    continue
                best = None
                for i, word in enumerate(expected):
                    similarity = self.similarity(word, heard[-1])
                    if similarity < .60:
                        continue
                    preceding = self.similarity(expected[i - 1], heard[-2]) if len(heard)>1 and i>0 else 0
                    average = (preceding + similarity) / 2
                    pair = preceding >= .60 and average >= .80
                    if p < current or (self.predicted and p == current):
                        # Two common words must not undo linear progress.
                        # A sung repeat needs a strong ordered third word.
                        third = self.similarity(expected[i-2], heard[-3]) if i>1 and len(heard)>2 else 0
                        pair = pair and third >= .72 and (third + preceding + similarity) / 3 >= .85
                    # A one-word start is only enabled by the ordered next
                    # expectation, never by global shabad search.
                    single = self.predicted and p == current + 1 and i == 0 and similarity >= .86
                    if not pair and not single:
                        continue
                    value = (int(pair), average if pair else similarity,
                             -abs(p - current - int(self.predicted)))
                    if best is None or value > best:
                        best = value
                if best is not None:
                    candidates.append((best, self.evidence[bani_id], bani_id == self.selected,
                                       -len(rows), bani_id, p, rows[p]))
        if not candidates:
            return None  # Leave switching to the existing shabad matcher.
        best, _, _, _, bani_id, p, row = max(candidates, key=lambda x: x[:6])
        self.selected = bani_id
        payload = self._payload(bani_id, p, best[1], len(tokens))
        return self.adopt(payload, transcript, message)
