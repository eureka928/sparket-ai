from __future__ import annotations

from enum import Enum


class League(str, Enum):
    NFL = "NFL"
    NBA = "NBA"
    MLB = "MLB"
    NHL = "NHL"
    SOCCER = "Soccer"


class SeasonType(str, Enum):
    PRE = "Preseason"
    REG = "Regular"
    POST = "Postseason"


class GameStatus(str, Enum):
    """
    All possible game statuses from SportsDataIO across NFL, NBA, MLB, NHL, Soccer.
    See: https://support.sportsdata.io/hc/en-us/articles/14287629964567-Process-Guide-Game-Status
    """
    # Common statuses (all sports)
    SCHEDULED = "Scheduled"
    IN_PROGRESS = "InProgress"
    FINAL = "Final"
    POSTPONED = "Postponed"
    CANCELED = "Canceled"
    SUSPENDED = "Suspended"
    DELAYED = "Delayed"
    FORFEIT = "Forfeit"
    # NHL/NBA overtime/shootout variants
    FINAL_OT = "F/OT"      # Final/Overtime (NBA, NHL)
    FINAL_SO = "F/SO"      # Final/Shootout (NHL only)
    # Soccer-specific statuses
    BREAK = "Break"                    # Halftime or other break
    AWARDED = "Awarded"                # Victory awarded (walkover/forfeit)
    EXTRA_TIME = "ExtraTime"           # Match in extra time (AET)
    PENALTY_SHOOTOUT = "PenaltyShootout"  # Match in penalty shootout
    # Soccer final variants
    FINAL_AET = "FinalAET"             # Final after extra time
    FINAL_PEN = "FinalPEN"             # Final after penalties


class MarketWindow(str, Enum):
    PRE = "pre"
    LIVE = "live"
    CLOSE = "close"


