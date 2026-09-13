import tidypyrs as tp


def test_map_columns():
    tf = tp.TibbleFrame({"a": [1, 2, 3, 4], "b": ["10", "20", "30", "40"]})

    actual = tf.map_columns("a", lambda s: s.shrink_dtype())

    expected = tp.TibbleFrame(
        {"a": [1, 2, 3, 4], "b": ["10", "20", "30", "40"]},
        schema={"a": tp.Int8, "b": tp.String}
    )

    assert actual.equals(expected)


def test_map_rows():
    tf = tp.TibbleFrame({"foo": [1, 2, 3], "bar": [-1, 5, 8]})

    actual = tf.map_rows(lambda t: (t[0] * 2, t[1] * 3))

    expected = tp.TibbleFrame(
        column_0=[2, 4, 6],
        column_1=[-3, 15, 24]
    )

    assert actual.equals(expected)
