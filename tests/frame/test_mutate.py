import tidypyrs as tp


def test_mutate_parallel():
    tf = tp.TibbleFrame(a=[1, 2])

    actual = tf.mutate(
        double_a=tp.col("a") * 2,
        triple_a=tp.col("a") * 3,
    )

    expected = tp.TibbleFrame(
        a=[1, 2],
        double_a=[2, 4],
        triple_a=[3, 6],
    )

    assert actual.equals(expected)


def test_mutate_sequential():
    tf = tp.TibbleFrame(a=[1, 2])

    actual = tf.mutate(
        double_a=tp.col("a") * 2,
        quadruple_a=tp.col("double_a") * 2,
        parallel=False,
    )

    expected = tp.TibbleFrame(
        a=[1, 2],
        double_a=[2, 4],
        quadruple_a=[4, 8],
    )

    assert actual.equals(expected)
