"""Conservative face-centroid tracking and directional doorway crossing."""
from dataclasses import dataclass, field
from math import hypot


@dataclass
class Track:
    point: tuple
    last_seen: float
    side: int = 0
    candidate: str | None = None
    votes: int = 0
    name: str = ''
    score: float = 0.0
    emitted: bool = False
    conflicted: bool = False


class EntryTracker:
    """Coordinates normalized to image size. Positive side means right/bottom."""
    def __init__(self, axis='x', line=.5, direction=1, hysteresis=.04,
                 max_distance=.15, max_age=1.0, confirmations=3):
        if axis not in ('x', 'y') or direction not in (-1, 1):
            raise ValueError('Invalid axis or direction')
        if not 0 < line < 1 or not 0 < hysteresis < min(line, 1-line):
            raise ValueError('Line/hysteresis must fit within the image')
        if confirmations < 1 or max_age <= 0 or max_distance <= 0:
            raise ValueError('Invalid tracking limits')
        self.axis, self.line, self.direction = axis, line, direction
        self.hysteresis, self.max_distance = hysteresis, max_distance
        self.max_age, self.confirmations = max_age, confirmations
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, now):
        """detections: [(normalized_center, (employee_id,name,score) | None)]."""
        self.tracks = {k: t for k, t in self.tracks.items()
                       if now - t.last_seen <= self.max_age}
        # Only associate mutually unique spatial candidates. Crowded/ambiguous
        # matches are discarded instead of risking an identity switch.
        links = {i: [k for k, t in self.tracks.items()
                     if hypot(p[0]-t.point[0], p[1]-t.point[1]) <= self.max_distance]
                 for i, (p, _) in enumerate(detections)}
        counts = {k: sum(k in ks for ks in links.values()) for k in self.tracks}
        events = []
        for i, (point, identity) in enumerate(detections):
            keys = links[i]
            if len(keys) == 1 and counts[keys[0]] == 1:
                t = self.tracks[keys[0]]
            elif keys:
                # Invalidate continuity on ambiguity; no crossing may span it.
                for k in keys:
                    self.tracks[k].conflicted = True
                continue
            else:
                t = Track(point, now)
                self.tracks[self.next_id] = t
                self.next_id += 1
            t.point, t.last_seen = point, now
            if identity is None:
                t.votes = 0
            if identity:
                employee_id, name, score = identity
                if t.candidate is None:
                    t.candidate, t.name = employee_id, name
                elif t.candidate != employee_id:
                    t.conflicted = True
                if not t.conflicted:
                    t.votes += 1
                    t.score = score
            coordinate = point[0 if self.axis == 'x' else 1]
            side = (-1 if coordinate < self.line-self.hysteresis else
                    1 if coordinate > self.line+self.hysteresis else 0)
            if side:
                if (t.side == -self.direction and side == self.direction
                        and not t.emitted):
                    # A crossing without confirmed identity cannot become an
                    # entry later while someone stands inside the office.
                    t.emitted = True
                    if t.votes >= self.confirmations and not t.conflicted:
                        events.append((t.candidate, t.name, t.score))
                t.side = side
        return events
