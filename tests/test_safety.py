"""安全层测试：坐标夹紧、风险关键词判定、关闭确认。"""

from agent.safety import Safety


def test_clamp_in_bounds():
    s = Safety({"safety": {"clamp_coords": True}})
    assert s.clamp(5, 5) == (5, 5)


def test_clamp_out_of_bounds():
    s = Safety({"safety": {"clamp_coords": True}})
    x, y = s.clamp(99999, -10)
    assert 0 <= x <= s.W and 0 <= y <= s.H


def test_clamp_disabled():
    s = Safety({"safety": {"clamp_coords": False}})
    assert s.clamp(5, 5) == (5, 5)


def test_is_risky_match():
    s = Safety({"safety": {"confirm_risky": True, "risk_keywords": ["delete", "删除"]}})
    assert s.is_risky({"action": "key", "key": "delete"}) is True
    assert s.is_risky({"action": "type", "text": "删除文件"}) is True


def test_is_risky_no_match():
    s = Safety({"safety": {"confirm_risky": True, "risk_keywords": ["delete"]}})
    assert s.is_risky({"action": "click", "x": 1, "y": 1}) is False


def test_is_risky_disabled():
    s = Safety({"safety": {"confirm_risky": False, "risk_keywords": ["delete"]}})
    assert s.is_risky({"action": "key", "key": "delete"}) is False
