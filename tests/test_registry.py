import pytest

from csi_comp.registry import REGISTRY, available, get, register


@pytest.fixture(autouse=True)
def _scratch_kind():
    REGISTRY.setdefault("_test", {})
    REGISTRY["_test"].clear()
    yield
    REGISTRY["_test"].clear()


def test_register_and_get():
    @register("_test", "foo")
    class Foo: ...

    assert get("_test", "foo") is Foo


def test_duplicate_registration_raises():
    @register("_test", "x")
    class A: ...

    with pytest.raises(KeyError):

        @register("_test", "x")
        class B: ...


def test_redecorating_same_class_is_idempotent():
    @register("_test", "y")
    class C: ...

    # Re-decorating the same class with the same name should not raise.
    register("_test", "y")(C)
    assert get("_test", "y") is C


def test_unknown_kind_raises():
    with pytest.raises(KeyError):
        register("nope_kind", "x")(object)


def test_missing_name_lists_available():
    @register("_test", "alpha")
    class _A: ...

    with pytest.raises(KeyError) as ei:
        get("_test", "beta")
    assert "alpha" in str(ei.value)


def test_available_listing():
    @register("_test", "b")
    class _B: ...

    @register("_test", "a")
    class _A: ...

    assert available("_test") == ["a", "b"]
