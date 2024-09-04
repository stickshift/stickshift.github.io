from stickshift import random_string


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
