import json
from unittest.mock import MagicMock, patch

from reco_nova.app import api_request, build_payload


def test_build_personalized_payload_is_minimal():
    assert build_payload("Personalized", 8, user_id=" user-1 ", age=30) == {
        "limit": 8,
        "user_id": "user-1",
    }


def test_build_discovery_payload_parses_context():
    payload = build_payload(
        "Discover",
        6,
        age=28,
        membership="Not specified",
        product_group="Accessories",
        session_items=" a, b, ,c ",
    )
    assert payload == {
        "limit": 6,
        "age": 28,
        "preferred_product_group": "Accessories",
        "session_article_ids": ["a", "b", "c"],
    }


@patch("reco_nova.app.urlopen")
def test_api_request_serializes_payload(mock_urlopen):
    response = MagicMock()
    response.read.return_value = json.dumps({"strategy": "test"}).encode()
    mock_urlopen.return_value.__enter__.return_value = response
    result = api_request("/recommend", {"limit": 4})
    request = mock_urlopen.call_args.args[0]
    assert result == {"strategy": "test"}
    assert json.loads(request.data) == {"limit": 4}
    assert request.get_method() == "POST"
