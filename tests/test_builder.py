import pytest
from nsp_sandbox.NSP_builder import NSPbuilder

version = "0.0.1"
def test_init():
    builder = NSPbuilder()
    assert builder.print_info, "Printing should be done by default."
    assert builder.version == version, "Version sould be " + version