"""Load tracking, metadata, and tactical tags into NumPy arrays.

The parsed tracking cache is stored in ``solution/cache/`` and is gitignored.
"""
import hashlib
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from games import GAME, GAMES, half_video_blocks, require  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parent / "cache"
FPS = GAMES[GAME]["fps"]
N_FRAMES = GAMES[GAME]["n_video_frames"]
PITCH = np.asarray(GAMES[GAME]["pitch"], dtype=float)  # metres, (length, width)
HOME, AWAY = 0, 1


@dataclass
class Tag:
    code: str
    start: int          # video frame, inclusive
    end: int            # video frame, exclusive
    team: int | None    # HOME / AWAY / None
    labels: dict = field(default_factory=dict)


@dataclass
class Match:
    players: np.ndarray        # (n, slots, 2) metres, NaN where untracked
    player_speed: np.ndarray   # (n, slots) m/s as reported by the tracker
    ball: np.ndarray           # (n, 2) metres, NaN where the ball is not tracked
    slot_team: np.ndarray      # (slots,) HOME / AWAY
    goalkeeper: np.ndarray     # (slots,) bool
    attack_sign: dict          # {half: (+1|-1 for HOME, +1|-1 for AWAY)} along pitch x
    halves: list               # [(half, first_frame, last_frame_inclusive)]
    tags: list                 # [Tag]

    @property
    def n(self):
        return len(self.ball)

    def half_of(self):
        """Per-frame half number, 0 where the frame is outside both graded halves."""
        out = np.zeros(self.n, np.int8)
        for half, first, last in self.halves:
            out[first:last + 1] = half
        return out


def _sha(path):
    """Return a short content hash used to key the parsed tracking cache."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _parse_raw(raw_path, n_slots):
    """Parse ``frame:x,y,s;x,y,s;...:bx,by,bz`` lines into one float array, fast."""
    text = Path(raw_path).read_text()
    flat = np.array(text.replace(":", ",").replace(";", ",").split(), dtype=object)
    width = 1 + 3 * n_slots + 3
    rows = [np.array(line.split(","), dtype=float) for line in flat]
    table = np.vstack(rows)
    if table.shape[1] != width:
        raise ValueError(f"expected {width} columns, got {table.shape[1]}")
    return table


def _metadata(meta_path):
    """Parse slot ownership, goalkeeper flags, and first-half directions."""
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    names = [team.findtext("Name") for team in teams]
    if names[0] != "Barcelona":
        raise ValueError(f"expected the home team (Barcelona) first, got {names}")
    team_index = {team.get("id"): i for i, team in enumerate(teams)}
    direction = {}
    for i, team in enumerate(teams):
        for param in team.iter("ProviderParameter"):
            if param.findtext("Name") == "attack_direction_first_half":
                direction[i] = 1 if param.findtext("Value") == "left_to_right" else -1
    players = list(root.iter("Player"))
    slot_team = np.array([team_index[p.get("teamId")] for p in players], dtype=np.int8)
    goalkeeper = np.zeros(len(players), bool)
    for i, player in enumerate(players):
        params = {p.findtext("Name"): p.findtext("Value") for p in player.iter("ProviderParameter")}
        goalkeeper[i] = (params.get("position_type") or "").casefold() == "goalkeeper"
    return names, slot_team, goalkeeper, direction


def _team_from_tag(text):
    """Map the provider's team spelling to the repository's Home/Away index."""
    value = (text or "").strip().casefold()
    if value in {"barcelona", "fc barcelona"}:
        return HOME
    if value in {"atletic club", "athletic club", "athletic"}:
        return AWAY
    return None


def load_tags(tag_path, n=N_FRAMES):
    """Load and time-align automatic tactical tags into sorted ``Tag`` records."""
    tags = []
    for event in ET.parse(tag_path).iter("instance"):
        code = (event.findtext("code") or "").strip().upper()
        labels = {(x.findtext("group") or "").strip(): (x.findtext("text") or "").strip()
                  for x in event.findall("label")}
        start = int(np.clip(round(float(event.findtext("start")) * FPS), 0, n))
        end = int(np.clip(round(float(event.findtext("end")) * FPS), 0, n))
        if end <= start:
            continue
        tags.append(Tag(code, start, end, _team_from_tag(labels.get("Team")), labels))
    tags.sort(key=lambda t: t.start)
    return tags


def load_match(data_dir=None):
    """Load the challenge match into one immutable, metre-space ``Match`` object."""
    meta, raw, tag_path = require(["atd_meta", "atd_raw", "smart_tagging"], data_dir=data_dir)
    names, slot_team, goalkeeper, direction = _metadata(meta)
    n_slots = len(slot_team)

    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"atd_{_sha(raw)}.npz"
    if cache.exists():
        table = np.load(cache)["table"]
    else:
        table = _parse_raw(raw, n_slots)
        np.savez_compressed(cache, table=table)

    if table.shape[0] != N_FRAMES or not np.array_equal(table[:, 0], np.arange(1, N_FRAMES + 1)):
        raise ValueError("ATD frame ids must be contiguous 1..N (video frame = id - 1)")
    body = table[:, 1:1 + 3 * n_slots].reshape(N_FRAMES, n_slots, 3)
    players = body[:, :, :2] * PITCH
    player_speed = body[:, :, 2]
    ball = table[:, 1 + 3 * n_slots:1 + 3 * n_slots + 2] * PITCH

    blocks = [(h, a, b) for h, a, b in half_video_blocks(GAME)]
    # Teams change ends at half-time.
    attack_sign = {1: (direction[HOME], direction[AWAY]),
                   2: (-direction[HOME], -direction[AWAY])}
    return Match(players, player_speed, ball, slot_team, goalkeeper, attack_sign, blocks,
                 load_tags(tag_path))
