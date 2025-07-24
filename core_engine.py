from dataclasses import dataclass, field
from typing import List, Dict, Union
import math
import uuid
# -- Symbolic Event Structures -- #
@dataclass
class SymbolicEvent:
    id: str
    drift: float
    resonance: float
    recurrence: int
@dataclass
class MemoryAnchor:
    id: str
    mass: float
    curvature: float
    compressed_from: List[str] = field(default_factory=list)
# -- Memory Mass & Curvature Engine -- #
class MemoryField:
    def init(self, threshold: float = 1.0):
        self.symbolic_events: List[SymbolicEvent] = []
        self.anchors: Dict[str, MemoryAnchor] = {}
        self.reentry_threshold = threshold
    def add_event(self, drift: float, resonance: float, recurrence: int):
        eid = str(uuid.uuid4())
        self.symbolic_events.append(SymbolicEvent(eid, drift, resonance, recurrence))
    def compute_mass(self, event: SymbolicEvent) -> float:
        return event.drift * event.resonance * event.recurrence
    def compute_curvature(self, mass: float) -> float:
        return math.sqrt(mass)  # Simplified curvature model
    def check_reentry(self, event: SymbolicEvent) -> bool:
        mass = self.compute_mass(event)
        curvature = self.compute_curvature(mass)
        return curvature > self.reentry_threshold
    def compress_collapse(self):
        grouped = {}
        for e in self.symbolic_events:
            key = f"{round(e.drift, 1)}:{round(e.resonance, 1)}"
            grouped.setdefault(key, []).append(e)
        for key, group in grouped.items():
            if len(group) < 2:
                continue
            total_mass = sum(self.compute_mass(e) for e in group)
            curvature = self.compute_curvature(total_mass)
            anchor_id = str(uuid.uuid4())
            self.anchors[anchor_id] = MemoryAnchor(
                id=anchor_id,
                mass=total_mass,
                curvature=curvature,
                compressed_from=[e.id for e in group]
            )
        # Optional: Clear compressed events
        self.symbolic_events = [e for group in grouped.values() if len(group) < 2 for e in group]
    def list_reentries(self) -> List[SymbolicEvent]:
        return [e for e in self.symbolic_events if self.check_reentry(e)]
    def dump_anchors(self) -> List[Dict[str, Union[str, float]]]:
        return [
            {
                "id": a.id,
                "mass": a.mass,
                "curvature": a.curvature,
                "from": a.compressed_from
            }
            for a in self.anchors.values()
        ]
