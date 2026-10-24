"""ToolAlpaca scoring regression tests, including evaluator false-negative fixes."""

from eval.scoring import extract_action_inputs, extract_actions, score_tooluse_response

GOLD = [
    {"Action": "getWeather", "Action_Input": '{"city": "Paris"}'},
    {"Action": "getTime", "Action_Input": '{"tz": "CET"}'},
]


def test_correct_response():
    response = 'Action: getWeather\nAction Input: {"city": "Paris"}\nAction: getTime\nAction Input: {"tz": "CET"}'
    score, details = score_tooluse_response(response, GOLD)
    assert score == 1 and details["correct"] is True


def test_action_order_does_not_matter():
    response = 'Action: getTime\nAction Input: {"tz": "CET"}\nAction: getWeather\nAction Input: {"city": "Paris"}'
    assert score_tooluse_response(response, GOLD)[0] == 1


def test_action_count_matters():
    response = 'Action: getWeather\nAction: getWeather\nAction: getTime\nAction Input: {"city": "Paris", "tz": "CET"}'
    score, details = score_tooluse_response(response, GOLD)
    assert score == 0 and not details["actions_match"] and details["inputs_match"]


def test_wrong_input_value():
    response = 'Action: getWeather\nAction Input: {"city": "Rome"}\nAction: getTime\nAction Input: {"tz": "CET"}'
    score, details = score_tooluse_response(response, GOLD)
    assert score == 0 and details["actions_match"] and not details["inputs_match"]


def test_malformed_json_blocks_are_skipped():
    assert extract_action_inputs('Action Input: {not json}\nAction Input: {"a": 1}') == {"a": 1}


def test_extract_actions():
    assert extract_actions("Action: a_b\nfoo\nAction:   c") == ["a_b", "c"]


def test_malformed_gold_input_is_ignored():
    gold = [{"Action": "x", "Action_Input": "not json"}]
    assert score_tooluse_response("Action: x", gold)[0] == 1


def test_empty_gold_and_empty_response():
    assert score_tooluse_response("no tools needed", [])[0] == 1


def test_nested_action_input_is_parsed_completely():
    response = (
        'Action: updateUserProfile\n'
        'Action Input: {"userId": "g-user123", "accessToken": "abcd1234", '
        '"profileData": {"profilePictureUrl": "https://example.com/new_image.jpg"}}'
    )
    gold = [
        {
            "Action": "updateUserProfile",
            "Action_Input": (
                '{"userId": "g-user123", "accessToken": "abcd1234", '
                '"profileData": {"profilePictureUrl": "https://example.com/new_image.jpg"}}'
            ),
        }
    ]
    score, details = score_tooluse_response(response, gold)
    assert score == 1
    assert details["pred_inputs"]["profileData"]["profilePictureUrl"] == "https://example.com/new_image.jpg"


def test_language_full_name_and_iso_code_are_equivalent():
    response = 'Action: getHolidaysByCountry\nAction Input: {"country": "Japan", "year": 2023, "language": "en"}'
    gold = [
        {
            "Action": "getHolidaysByCountry",
            "Action_Input": '{"country": "Japan", "year": 2023, "language": "English"}',
        }
    ]
    score, details = score_tooluse_response(response, gold)
    assert score == 1
    assert details["inputs_match"] is True
    assert details["pred_inputs_normalized"]["language"] == "en"
    assert details["gt_inputs_normalized"]["language"] == "en"


def test_spanish_full_name_and_iso_code_are_equivalent():
    response = 'Action: searchVerses\nAction Input: {"text": "amor", "language": "es"}'
    gold = [{"Action": "searchVerses", "Action_Input": '{"text": "amor", "language": "Spanish"}'}]
    assert score_tooluse_response(response, gold)[0] == 1


def test_common_language_aliases_are_supported_without_touching_other_strings():
    response = 'Action: translate\nAction Input: {"language": "French", "text": "English"}'
    gold = [{"Action": "translate", "Action_Input": '{"language": "fr", "text": "English"}'}]
    assert score_tooluse_response(response, gold)[0] == 1

    # Only language-valued fields are canonicalized: free text is still exact.
    wrong_text = 'Action: translate\nAction Input: {"language": "fr", "text": "en"}'
    assert score_tooluse_response(wrong_text, gold)[0] == 0


def test_comma_separated_symbol_whitespace_is_ignored():
    response = 'Action: getExchangeRates\nAction Input: {"base": "GBP", "symbols": "USD, EUR, JPY"}'
    gold = [
        {
            "Action": "getExchangeRates",
            "Action_Input": '{"base": "GBP", "symbols": "USD,EUR,JPY"}',
        }
    ]
    score, details = score_tooluse_response(response, gold)
    assert score == 1
    assert details["pred_inputs"]["symbols"] == "USD, EUR, JPY"
    assert details["pred_inputs_normalized"]["symbols"] == "USD,EUR,JPY"


def test_comma_whitespace_in_free_text_is_not_normalized():
    response = 'Action: x\nAction Input: {"comment": "hello, world"}'
    gold = [{"Action": "x", "Action_Input": '{"comment": "hello,world"}'}]
    assert score_tooluse_response(response, gold)[0] == 0
