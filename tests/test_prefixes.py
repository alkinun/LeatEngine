from leatengine.prefixes import Prefix, PrefixStore


def prefix(owner: str, cached: list[int], nbytes: int = 10) -> Prefix:
    return Prefix(owner, cached, [], len(cached), [], None, nbytes)


def test_least_recently_used_goes_first():
    store = PrefixStore(30)
    a, b, c = prefix("x", [1, 2]), prefix("x", [3, 4]), prefix("y", [5])
    for p in (a, b, c):
        store.add(p)
    store.take(a)  # used again: b is now the least recently used
    store.add(d := prefix("y", [6], 10))
    assert store.of("x") == [a] and store.of("y") == [d, c] and store.used == 30


def test_a_prefix_replaces_those_it_holds():
    # of the same owner only, and not those that hold more than it
    store = PrefixStore(100)
    for p in (short := prefix("x", [1, 2]), other := prefix("y", [1, 2]), prefix("x", [1, 3])):
        store.add(p)
    store.add(longer := prefix("x", [1, 2, 3]))
    assert short not in store.of("x") and longer in store.of("x") and other in store.of("y")
    assert store.holds("x", [1, 2]) and store.holds("x", [1, 2, 3]) and not store.holds("x", [2])
    assert not store.holds("z", [1]) and store.used == 30


def test_too_large_a_prefix_is_not_kept():
    store = PrefixStore(15)
    store.add(prefix("x", [1], 20))
    assert len(store) == 0 and store.used == 0
    store.add(prefix("x", [1], 15))
    store.clear()
    assert len(store) == 0 and store.used == 0
