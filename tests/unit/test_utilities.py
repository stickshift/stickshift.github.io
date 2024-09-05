from pathlib import Path

from stickshift import random_string, shell, take


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


def test_take():
    #
    # Givens
    #

    # A list of alternating keys and values
    data = [1, "one", 2, "two"]

    #
    # Whens
    #

    # I create an iterator to process values 2 at a time
    it = take(2, data)

    #
    # Thens
    #

    # First pair should be (1, "one")
    assert next(it) == (1, "one")

    # Second pair should be (2, "two")
    assert next(it) == (2, "two")

    # There shouldn't be any more pairs
    assert next(it, None) is None

