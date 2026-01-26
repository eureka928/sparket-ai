from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import pytest

from sparket.providers.sportsdataio.config import LeagueCode, LeagueConfig, SportsDataIOConfig
from sparket.providers.sportsdataio.types import Game, GameOdds, GameOddsSet
from sparket.validator.services import SportsDataIngestor, TrackedEvent


class FakeClient:
    def __init__(self, games: Any, odds: Any):
        self._games_by_league: Dict[str, List[Game]] = {}
        self._games_default: List[Game] = []
        if isinstance(games, dict):
            for key, value in games.items():
                league_key = self._normalize_league_key(key)
                entries = value if isinstance(value, (list, tuple)) else [value]
                self._games_by_league[league_key] = list(entries)
        elif isinstance(games, (list, tuple)):
            self._games_default = list(games)
        elif games is not None:
            self._games_default = [games]

        self._history: Dict[tuple[str, int] | int, GameOddsSet] = {}
        if isinstance(odds, dict):
            for key, value in odds.items():
                if isinstance(key, tuple) and len(key) == 2:
                    league_key = self._normalize_league_key(key[0])
                    self._history[(league_key, int(key[1]))] = value
                else:
                    self._history[int(key)] = value
        elif isinstance(odds, (list, tuple)):
            for item in odds:
                self._history[int(item.game_id)] = item
        elif odds is not None:
            self._history[int(odds.game_id)] = odds
        self.schedule_requests = 0
        self.odds_requests = 0

    async def close(self) -> None:  # pragma: no cover - not used
        return None

    async def fetch_team_catalog(self, league_config):  # pragma: no cover - skipped
        return []

    async def fetch_schedule_window(self, league_config, start_date, end_date):
        self.schedule_requests += 1
        games = self._games_by_league.get(self._normalize_league_key(league_config), self._games_default)
        return list(games)

    async def fetch_schedule_season(self, league_config, season_code, season_type=None):
        self.schedule_requests += 1
        games = self._games_by_league.get(self._normalize_league_key(league_config), self._games_default)
        return list(games)

    async def fetch_line_history(self, league_config, game_id):
        self.odds_requests += 1
        league_key = self._normalize_league_key(league_config)
        return self._history.get((league_key, int(game_id))) or self._history.get(int(game_id))

    def _normalize_league_key(self, league_config) -> str:
        if isinstance(league_config, LeagueConfig):
            return league_config.code.value
        if isinstance(league_config, LeagueCode):
            return league_config.value
        if isinstance(league_config, str):
            try:
                return LeagueCode(league_config).value
            except Exception:
                return league_config.lower()
        return str(league_config).lower()


class StubDatabase:
    async def read(self, *_, **__):
        return []

    async def write(self, *_, **__):
        return []


def base_league_config() -> LeagueConfig:
    return LeagueConfig(
        code=LeagueCode.NFL,
        league_code="nfl",
        sport_code="football",
        schedule_url="https://example.com/schedule/{DATE}",
        odds_url="https://example.com/odds/{GAMEID}",
        delta_url=None,
        teams_url="https://example.com/teams",
        schedule_refresh_minutes=60,
        odds_refresh_minutes=15,
        hot_odds_refresh_minutes=5,
        delta_minutes=10,
        hot_delta_minutes=2,
        track_days_ahead=7,
    )


def test_run_once_invokes_persist(monkeypatch):
    asyncio.run(_run_once_invokes_persist(monkeypatch))


async def _run_once_invokes_persist(monkeypatch):
    now = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    start_dt = now + timedelta(days=2)
    game = Game.model_validate(
        {
            "GameID": 1,
            "Season": 2024,
            "SeasonType": "Regular",
            "Week": 1,
            "Date": start_dt.isoformat(),
            "HomeTeam": "NE",
            "AwayTeam": "NYJ",
        }
    )
    odds = GameOdds.model_validate(
        {
            "GameID": 1,
            "Sportsbook": "TestBook",
            "Updated": now.isoformat(),
            "MoneyLineHome": -110,
            "MoneyLineAway": 120,
            "PointSpread": -3.5,
            "PointSpreadHome": -110,
            "PointSpreadAway": -110,
            "OverUnder": 45.5,
            "OverPayout": -110,
            "UnderPayout": -105,
        }
    )
    odds_set = GameOddsSet(game_id=1, pregame=[odds])

    league_cfg = LeagueConfig(
        code=LeagueCode.NFL,
        league_code="nfl",
        sport_code="football",
        schedule_url="https://example.com/schedule/{DATE}",
        odds_url="https://example.com/odds/{GAMEID}",
        delta_url=None,
        teams_url="https://example.com/teams",
        schedule_refresh_minutes=60,
        odds_refresh_minutes=15,
        hot_odds_refresh_minutes=5,
        delta_minutes=10,
        hot_delta_minutes=2,
        track_days_ahead=7,
    )
    config = SportsDataIOConfig(leagues=[league_cfg])
    client = FakeClient(game, odds_set)
    ingestor = SportsDataIngestor(database=StubDatabase(), client=client, config=config)

    # Pre-load league metadata to avoid DB dependency
    for state in ingestor.leagues.values():
        state.league_id = 10
        state.team_index = {"NE": 1, "NYJ": 2}

    async def fake_ensure_event_for_sdio(database, event_row):
        return int(event_row["ext_ref"]["sportsdataio"]["GameID"]), event_row["start_time_utc"]

    monkeypatch.setattr(
        "sparket.validator.services.sportsdata_ingestor.ensure_event_for_sdio",
        fake_ensure_event_for_sdio,
    )

    persist_calls: list[int] = []

    async def fake_persist(self, state, tracked, odds_set):
        persist_calls.append(tracked.game_id)
        return tracked.start_time

    monkeypatch.setattr(
        SportsDataIngestor,
        "_persist_odds",
        fake_persist,
    )

    await ingestor.run_once(now=now)

    assert len(ingestor.tracked_events) == 1
    assert persist_calls == [1]
    assert client.schedule_requests == 1
    assert client.odds_requests == 1


def test_snapshot_interval_speeds_up():
    now = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    game = Game.model_validate(
        {
            "GameID": 9,
            "Season": 2024,
            "SeasonType": "Regular",
            "Week": 1,
            "Date": now.isoformat(),
            "HomeTeam": "CHI",
            "AwayTeam": "GB",
        }
    )
    odds = GameOdds.model_validate(
        {
            "GameID": 9,
            "Sportsbook": "TestBook",
            "Updated": now.isoformat(),
            "MoneyLineHome": -110,
            "MoneyLineAway": 120,
            "PointSpread": -3.5,
            "PointSpreadHome": -110,
            "PointSpreadAway": -110,
            "OverUnder": 45.5,
            "OverPayout": -110,
            "UnderPayout": -105,
        }
    )
    odds_set = GameOddsSet(game_id=9, pregame=[odds])

    league_cfg = LeagueConfig(
        code=LeagueCode.NFL,
        league_code="nfl",
        sport_code="football",
        schedule_url="https://example.com/schedule/{DATE}",
        odds_url="https://example.com/odds/{GAMEID}",
        delta_url=None,
        teams_url="https://example.com/teams",
    )
    config = SportsDataIOConfig(leagues=[league_cfg])
    ingestor = SportsDataIngestor(database=StubDatabase(), client=FakeClient(game, odds_set), config=config)
    state = SimpleNamespace(config=league_cfg)

    tracked_far = TrackedEvent(
        league_code=LeagueCode.NFL,
        game_id=1,
        event_id=1,
        start_time=now + timedelta(days=3),
    )
    tracked_warm = TrackedEvent(
        league_code=LeagueCode.NFL,
        game_id=2,
        event_id=2,
        start_time=now + timedelta(hours=3),
    )
    tracked_hot = TrackedEvent(
        league_code=LeagueCode.NFL,
        game_id=3,
        event_id=3,
        start_time=now + timedelta(minutes=30),
    )

    far_interval = ingestor._next_snapshot_interval(state, tracked_far, now)
    warm_interval = ingestor._next_snapshot_interval(state, tracked_warm, now)
    hot_interval = ingestor._next_snapshot_interval(state, tracked_hot, now)

    assert far_interval > warm_interval > hot_interval


def test_schedule_tracks_only_window(monkeypatch):
    asyncio.run(_schedule_tracks_only_window(monkeypatch))


async def _schedule_tracks_only_window(monkeypatch):
    now = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    near = Game.model_validate(
        {
            "GameID": 50,
            "Season": 2025,
            "SeasonType": "Regular",
            "Date": (now + timedelta(hours=6)).isoformat(),
            "HomeTeam": "KC",
            "AwayTeam": "BUF",
        }
    )
    far = Game.model_validate(
        {
            "GameID": 51,
            "Season": 2025,
            "SeasonType": "Regular",
            "Date": (now + timedelta(days=20)).isoformat(),
            "HomeTeam": "DAL",
            "AwayTeam": "NYG",
        }
    )
    odds_set = GameOddsSet(game_id=50, pregame=[])
    client = FakeClient([near, far], odds_set)
    config = SportsDataIOConfig(leagues=[base_league_config()])
    ingestor = SportsDataIngestor(database=StubDatabase(), client=client, config=config)
    state = next(iter(ingestor.leagues.values()))
    state.league_id = 22
    state.team_index = {"KC": 1, "BUF": 2, "DAL": 3, "NYG": 4}

    async def fake_ensure_event(database, event_row):
        gid = event_row["ext_ref"]["sportsdataio"]["GameID"]
        return gid, event_row["start_time_utc"]

    monkeypatch.setattr(
        "sparket.validator.services.sportsdata_ingestor.ensure_event_for_sdio",
        fake_ensure_event,
    )

    inserted = await ingestor._refresh_schedule(state, now)
    assert inserted == 2
    assert (LeagueCode.NFL, 50) in ingestor.tracked_events
    assert (LeagueCode.NFL, 51) not in ingestor.tracked_events


def test_tracks_games_across_leagues(monkeypatch):
    asyncio.run(_tracks_games_across_leagues(monkeypatch))


async def _tracks_games_across_leagues(monkeypatch):
    now = datetime(2025, 1, 5, 15, 0, tzinfo=timezone.utc)
    nba_game = Game.model_validate(
        {
            "GameID": 77,
            "Season": 2024,
            "SeasonType": "Regular",
            "Date": (now + timedelta(hours=4)).isoformat(),
            "HomeTeam": "LAL",
            "AwayTeam": "BOS",
        }
    )
    nhl_game = Game.model_validate(
        {
            "GameID": 77,
            "Season": 2024,
            "SeasonType": "Regular",
            "Date": (now + timedelta(hours=5)).isoformat(),
            "HomeTeam": "NYR",
            "AwayTeam": "MTL",
        }
    )
    nba_odds = GameOdds.model_validate(
        {
            "GameID": nba_game.game_id,
            "Sportsbook": "HoopsBook",
            "Updated": now.isoformat(),
            "MoneyLineHome": -120,
            "MoneyLineAway": 110,
        }
    )
    nhl_odds = GameOdds.model_validate(
        {
            "GameID": nhl_game.game_id,
            "Sportsbook": "IceBook",
            "Updated": now.isoformat(),
            "MoneyLineHome": -130,
            "MoneyLineAway": 115,
        }
    )
    nba_cfg = LeagueConfig(
        code=LeagueCode.NBA,
        league_code="nba",
        sport_code="basketball",
        schedule_url="https://example.com/nba/schedule/{DATE}",
        odds_url="https://example.com/nba/odds/{GAMEID}",
        teams_url="https://example.com/nba/teams",
        track_days_ahead=3,
    )
    nhl_cfg = LeagueConfig(
        code=LeagueCode.NHL,
        league_code="nhl",
        sport_code="hockey",
        schedule_url="https://example.com/nhl/schedule/{DATE}",
        odds_url="https://example.com/nhl/odds/{GAMEID}",
        teams_url="https://example.com/nhl/teams",
        track_days_ahead=3,
    )
    config = SportsDataIOConfig(leagues=[nba_cfg, nhl_cfg])
    client = FakeClient(
        games={LeagueCode.NBA: [nba_game], LeagueCode.NHL: [nhl_game]},
        odds={
            (LeagueCode.NBA, nba_game.game_id): GameOddsSet(game_id=nba_game.game_id, pregame=[nba_odds]),
            (LeagueCode.NHL, nhl_game.game_id): GameOddsSet(game_id=nhl_game.game_id, pregame=[nhl_odds]),
        },
    )
    ingestor = SportsDataIngestor(database=StubDatabase(), client=client, config=config)
    for code, state in ingestor.leagues.items():
        state.league_id = 100 if code == LeagueCode.NBA else 200
        if code == LeagueCode.NBA:
            state.team_index = {"LAL": 1, "BOS": 2}
        else:
            state.team_index = {"NYR": 3, "MTL": 4}

    async def fake_ensure_event(database, event_row):
        gid = event_row["ext_ref"]["sportsdataio"]["GameID"]
        return gid + event_row["league_id"], event_row["start_time_utc"]

    async def fake_persist(self, state, tracked, odds_set):
        return tracked.start_time

    monkeypatch.setattr(
        "sparket.validator.services.sportsdata_ingestor.ensure_event_for_sdio",
        fake_ensure_event,
    )
    monkeypatch.setattr(SportsDataIngestor, "_persist_odds", fake_persist)

    await ingestor.run_once(now=now)
    assert len(ingestor.tracked_events) == 2
    tracked_leagues = {tracked.league_code for tracked in ingestor.tracked_events.values()}
    assert tracked_leagues == {LeagueCode.NBA, LeagueCode.NHL}


def test_persist_odds_inserts_quotes(monkeypatch):
    asyncio.run(_persist_odds_inserts_quotes(monkeypatch))


async def _persist_odds_inserts_quotes(monkeypatch):
    now = datetime(2025, 1, 2, 15, 0, tzinfo=timezone.utc)
    game = Game.model_validate(
        {
            "GameID": 7,
            "Season": 2024,
            "SeasonType": "Regular",
            "Week": 2,
            "Date": now.isoformat(),
            "HomeTeam": "KC",
            "AwayTeam": "BUF",
        }
    )
    odds = GameOdds.model_validate(
        {
            "GameID": 7,
            "Sportsbook": "TestBook",
            "Updated": now.isoformat(),
            "MoneyLineHome": -130,
            "MoneyLineAway": 110,
            "PointSpread": -3.0,
            "PointSpreadHome": -115,
            "PointSpreadAway": -105,
            "OverUnder": 47.5,
            "OverPayout": -110,
            "UnderPayout": -110,
        }
    )
    odds_set = GameOddsSet(game_id=7, pregame=[odds])

    league_cfg = LeagueConfig(
        code=LeagueCode.NFL,
        league_code="nfl",
        sport_code="football",
        schedule_url="https://example.com/schedule/{DATE}",
        odds_url="https://example.com/odds/{GAMEID}",
        delta_url=None,
        teams_url="https://example.com/teams",
    )
    config = SportsDataIOConfig(leagues=[league_cfg])
    ingestor = SportsDataIngestor(database=StubDatabase(), client=FakeClient(game, odds_set), config=config)

    tracked = TrackedEvent(
        league_code=LeagueCode.NFL,
        game_id=7,
        event_id=700,
        start_time=now + timedelta(hours=1),
    )

    async def fake_ensure_market(database, market_row, *, event_id):
        return 500 + event_id

    captured_quotes = []

    async def fake_insert_provider_quotes(*, database, quotes):
        batch = list(quotes)
        captured_quotes.append(batch)
        return len(batch)

    monkeypatch.setattr(
        "sparket.validator.services.sportsdata_ingestor.ensure_market",
        fake_ensure_market,
    )
    monkeypatch.setattr(
        "sparket.validator.services.sportsdata_ingestor.insert_provider_quotes",
        fake_insert_provider_quotes,
    )

    state = next(iter(ingestor.leagues.values()))
    result = await ingestor._persist_odds(state, tracked, odds_set)

    assert isinstance(result, datetime)
    assert result == now
    assert captured_quotes
    assert sum(len(batch) for batch in captured_quotes) >= 4


def test_history_filters_duplicates(monkeypatch):
    asyncio.run(_history_filters_duplicates(monkeypatch))


async def _history_filters_duplicates(monkeypatch):
    now = datetime(2025, 1, 2, 15, 0, tzinfo=timezone.utc)
    game = Game.model_validate(
        {
            "GameID": 8,
            "Season": 2024,
            "SeasonType": "Regular",
            "Week": 2,
            "Date": (now + timedelta(hours=5)).isoformat(),
            "HomeTeam": "KC",
            "AwayTeam": "BUF",
        }
    )
    odds_a = GameOdds.model_validate(
        {
            "GameID": 8,
            "GameOddId": 1,
            "Sportsbook": "TestBook",
            "Updated": (now - timedelta(hours=2)).isoformat(),
            "MoneyLineHome": -120,
            "MoneyLineAway": 110,
        }
    )
    odds_b = GameOdds.model_validate(
        {
            "GameID": 8,
            "GameOddId": 2,
            "Sportsbook": "TestBook",
            "Updated": (now - timedelta(hours=1)).isoformat(),
            "MoneyLineHome": -125,
            "MoneyLineAway": 115,
        }
    )
    odds_set = GameOddsSet(game_id=8, pregame=[odds_a, odds_b])
    league_cfg = base_league_config()
    config = SportsDataIOConfig(leagues=[league_cfg])
    ingestor = SportsDataIngestor(database=StubDatabase(), client=FakeClient(game, odds_set), config=config)

    state = next(iter(ingestor.leagues.values()))
    tracked = TrackedEvent(
        league_code=LeagueCode.NFL,
        game_id=8,
        event_id=800,
        start_time=now + timedelta(hours=4),
    )

    async def fake_ensure_market(database, market_row, *, event_id):
        return 500 + event_id

    async def fake_insert_provider_quotes(*, database, quotes):
        return len(list(quotes))

    monkeypatch.setattr(
        "sparket.validator.services.sportsdata_ingestor.ensure_market",
        fake_ensure_market,
    )
    monkeypatch.setattr(
        "sparket.validator.services.sportsdata_ingestor.insert_provider_quotes",
        fake_insert_provider_quotes,
    )

    first = await ingestor._persist_odds(state, tracked, odds_set)
    assert isinstance(first, datetime)
    seen = set(tracked.seen_odd_ids)
    assert seen == {1, 2}
    watermark = tracked.last_history_ts

    second = await ingestor._persist_odds(state, tracked, odds_set)
    assert second is None
    assert tracked.seen_odd_ids == seen
    assert tracked.last_history_ts == watermark


def test_season_schedule_mode(monkeypatch):
    asyncio.run(_season_schedule_mode(monkeypatch))


async def _season_schedule_mode(monkeypatch):
    now = datetime(2025, 7, 1, 12, 0, tzinfo=timezone.utc)
    game = Game.model_validate(
        {
            "GameID": 100,
            "Season": 2025,
            "SeasonType": "Regular",
            "Week": 1,
            "Date": (now + timedelta(days=2)).isoformat(),
            "HomeTeam": "DAL",
            "AwayTeam": "NYG",
        }
    )
    odds = GameOdds.model_validate(
        {
            "GameID": 100,
            "Sportsbook": "TestBook",
            "Updated": now.isoformat(),
            "MoneyLineHome": -110,
            "MoneyLineAway": 110,
        }
    )
    odds_set = GameOddsSet(game_id=100, pregame=[odds])
    league_cfg = LeagueConfig(
        code=LeagueCode.NFL,
        league_code="nfl",
        sport_code="football",
        schedule_url="https://example.com/nfl/scores/json/SchedulesBasic/{SEASON}",
        odds_url="https://example.com/odds/{GAMEID}",
        teams_url="https://example.com/teams",
        schedule_mode="season",
        season_format="{year}{season_type}",
        season_type="REG",
    )
    config = SportsDataIOConfig(leagues=[league_cfg])
    client = FakeClient([game], odds_set)
    ingestor = SportsDataIngestor(database=StubDatabase(), client=client, config=config)

    state = next(iter(ingestor.leagues.values()))
    state.league_id = 30
    state.team_index = {"DAL": 1, "NYG": 2}

    async def fake_ensure_event_for_sdio(database, event_row):
        return int(event_row.get("ext_ref", {}).get("sportsdataio", {}).get("GameID", 0)), event_row["start_time_utc"]

    monkeypatch.setattr(
        "sparket.validator.services.sportsdata_ingestor.ensure_event_for_sdio",
        fake_ensure_event_for_sdio,
    )

    async def fake_persist(self, state, tracked, odds_set):
        return tracked.start_time

    monkeypatch.setattr(SportsDataIngestor, "_persist_odds", fake_persist)

    await ingestor.run_once(now=now)
    assert client.schedule_requests == 1
    assert len(ingestor.tracked_events) == 1


def test_snapshot_cache_reuses_fetch(monkeypatch):
    asyncio.run(_snapshot_cache_reuses_fetch(monkeypatch))


async def _snapshot_cache_reuses_fetch(monkeypatch):
    now = datetime(2025, 1, 3, 12, 0, tzinfo=timezone.utc)
    game = Game.model_validate(
        {
            "GameID": 11,
            "Season": 2024,
            "SeasonType": "Regular",
            "Week": 3,
            "Date": (now + timedelta(days=1)).isoformat(),
            "HomeTeam": "DAL",
            "AwayTeam": "SF",
        }
    )
    odds = GameOdds.model_validate(
        {
            "GameID": 11,
            "Sportsbook": "TestBook",
            "Updated": now.isoformat(),
            "MoneyLineHome": -115,
            "MoneyLineAway": 105,
        }
    )
    odds_set = GameOddsSet(game_id=11, pregame=[odds])

    league_cfg = LeagueConfig(
        code=LeagueCode.NFL,
        league_code="nfl",
        sport_code="football",
        schedule_url="https://example.com/schedule/{DATE}",
        odds_url="https://example.com/odds/{GAMEID}",
        delta_url=None,
        teams_url="https://example.com/teams",
    )
    config = SportsDataIOConfig(leagues=[league_cfg])
    client = FakeClient([game], odds_set)
    ingestor = SportsDataIngestor(database=StubDatabase(), client=client, config=config)

    state = next(iter(ingestor.leagues.values()))
    state.league_id = 20
    tracked = TrackedEvent(
        league_code=LeagueCode.NFL,
        game_id=11,
        event_id=1100,
        start_time=now + timedelta(days=1, hours=2),
    )
    ingestor.tracked_events = {(LeagueCode.NFL, 11): tracked}

    async def fake_persist(self, state, tracked, odds_set):
        return tracked.start_time

    monkeypatch.setattr(SportsDataIngestor, "_persist_odds", fake_persist)

    await ingestor._refresh_odds(state, now)
    assert client.odds_requests == 1

    tracked.next_snapshot_at = datetime.min.replace(tzinfo=timezone.utc)
    await ingestor._refresh_odds(state, now + timedelta(minutes=1))
    assert client.odds_requests == 1


def test_post_start_finalizes_and_records_closing(monkeypatch):
    asyncio.run(_post_start_finalizes_and_records_closing(monkeypatch))


async def _post_start_finalizes_and_records_closing(monkeypatch):
    now = datetime(2025, 2, 1, 12, 0, tzinfo=timezone.utc)
    league_cfg = base_league_config()
    config = SportsDataIOConfig(leagues=[league_cfg])
    ingestor = SportsDataIngestor(database=StubDatabase(), client=FakeClient([], []), config=config)
    state = next(iter(ingestor.leagues.values()))
    state.league_id = 42
    tracked = TrackedEvent(
        league_code=LeagueCode.NFL,
        game_id=900,
        event_id=9000,
        start_time=now - timedelta(minutes=5),
    )
    ingestor.tracked_events = {(LeagueCode.NFL, tracked.game_id): tracked}

    closing_calls: list[datetime] = []

    async def fake_record(self, state, tracked, ts):
        closing_calls.append(ts)

    async def fake_resolve(self, state, tracked, now):
        odds = GameOdds.model_validate(
            {
                "GameID": tracked.game_id,
                "Sportsbook": "TestBook",
                "Updated": now.isoformat(),
            }
        )
        return GameOddsSet(game_id=tracked.game_id, pregame=[odds])

    async def fake_persist(self, state, tracked, odds_set):
        return now

    monkeypatch.setattr(SportsDataIngestor, "_record_closing_snapshot", fake_record)
    monkeypatch.setattr(SportsDataIngestor, "_resolve_line_history", fake_resolve)
    monkeypatch.setattr(SportsDataIngestor, "_persist_odds", fake_persist)

    metrics = {"snapshot_attempts": 0, "snapshot_success": 0, "snapshot_missed": 0}
    await ingestor._capture_snapshot(state, tracked, now, metrics=metrics)
    assert tracked.post_start_polls_remaining == 1
    assert closing_calls == [now]
    assert tracked.closing_captured is True

    await ingestor._capture_snapshot(state, tracked, now + timedelta(minutes=1), metrics=metrics)
    assert (LeagueCode.NFL, tracked.game_id) not in ingestor.tracked_events

