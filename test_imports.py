"""
Ship blocker test: Verify core scanner modules can be imported directly
without raising NameErrors due to typing or unimported forward references.
"""

def test_imports():
    import sp500_scanner
    import scanner_service.pipeline
    import scanner_service.api
    import scanner_service.alerts
    assert True
