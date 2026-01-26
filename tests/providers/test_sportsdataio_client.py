import asyncio
from datetime import date

from sparket.providers.sportsdataio.client import SportsDataIOClient, format_sdio_date
from sparket.providers.sportsdataio.config import LeagueCode, LeagueConfig, SportsDataIOConfig


def test_format_sdio_date_capitalizes_month():
    assert format_sdio_date(date(2025, 11, 19)) == "2025-NOV-19"
    assert format_sdio_date(date(2025, 1, 5)) == "2025-JAN-05"


def _league_cfg() -> LeagueConfig:
    return LeagueConfig(
        code=LeagueCode.NFL,
        league_code="nfl",
        sport_code="football",
        schedule_url="https://example.com/nfl/scores/json/SchedulesBasic/{SEASON}",
        odds_url="https://example.com/nfl/odds/json/GameOddsLineMovement/{GAMEID}",
        teams_url="https://example.com/nfl/teams",
        schedule_mode="season",
        season_format="{year}{season_type}",
        season_type="REG",
    )


def test_fetch_line_history_parses_single_payload(monkeypatch):
    asyncio.run(_assert_fetch_line_history_parses_single_payload(monkeypatch))


async def _assert_fetch_line_history_parses_single_payload(monkeypatch):
    league_cfg = _league_cfg()
    client = SportsDataIOClient(api_key="test", config=SportsDataIOConfig(leagues=[league_cfg]))

    async def fake_get_json(self, url):
        return {
            "GameID": 321,
            "PregameOdds": [
                {
                    "GameID": 321,
                    "GameOddId": 99,
                    "Sportsbook": "TestBook",
                    "Updated": "2025-01-01T12:00:00Z",
                    "MoneyLineHome": -110,
                    "MoneyLineAway": 100,
                }
            ],
        }

    monkeypatch.setattr(SportsDataIOClient, "_get_json", fake_get_json, raising=False)
    try:
        odds_set = await client.fetch_line_history(league_cfg, 321)
    finally:
        await client.close()

    assert odds_set is not None
    assert odds_set.game_id == 321
    assert odds_set.pregame
    assert odds_set.pregame[0].game_odd_id == 99


def test_fetch_line_history_handles_list_payload(monkeypatch):
    asyncio.run(_assert_fetch_line_history_handles_list_payload(monkeypatch))


async def _assert_fetch_line_history_handles_list_payload(monkeypatch):
    league_cfg = _league_cfg()
    client = SportsDataIOClient(api_key="test", config=SportsDataIOConfig(leagues=[league_cfg]))

    async def fake_get_json(self, url):
        return [
            {
                "GameID": 654,
                "PregameOdds": [
                    {
                        "GameID": 654,
                        "Sportsbook": "ListBook",
                        "Updated": "2025-01-02T15:00:00Z",
                        "MoneyLineHome": -105,
                        "MoneyLineAway": -105,
                    }
                ],
            }
        ]

    monkeypatch.setattr(SportsDataIOClient, "_get_json", fake_get_json, raising=False)
    try:
        odds_set = await client.fetch_line_history(league_cfg, 654)
    finally:
        await client.close()

    assert odds_set is not None
    assert odds_set.game_id == 654
    assert len(odds_set.pregame) == 1


