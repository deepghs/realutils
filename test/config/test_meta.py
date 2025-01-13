import pytest

from realutils.config.meta import __TITLE__


@pytest.mark.unittest
class TestConfigMeta:
    def test_title(self):
        assert __TITLE__ == 'realutils'
