from pathlib import Path

from stickshift import random_string, shell


def test_random_string():
    #
    # Whens
    #

    # I generate random string of length 16
    s = random_string(length=16)

    #
    # Thens
    #

    # s should be a string of length 16
    assert isinstance(s, str)
    assert len(s) == 16


def test_shell():
    #
    # Whens
    #

    # I list contents of /
    results = shell(f"ls /")

    # I split results on newlines
    results = results.splitlines()

    #
    # Thens
    #

    # results should include etc
    assert "etc" in results
