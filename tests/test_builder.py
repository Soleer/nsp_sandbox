import pytest
from nsp_sandbox.NSP_builder import NSPbuilder

def test_init():
    builder = NSPbuilder()
    assert builder.print_info, "Printing should be done by default."