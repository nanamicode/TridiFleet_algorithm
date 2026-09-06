from tridifleet.sim.audit import AuditLog


def test_audit_chain_detects_consistent_append_sequence():
    audit = AuditLog(":memory:")
    audit.start_run("2026-09-05T10:00:00-03:00", {"seed": 42})
    audit.append("context", "2026-09-05T10:01:00-03:00", {"reach": 10})
    audit.append("decision", "2026-09-05T10:01:00-03:00", {"ad": "A"})
    audit.append("feedback", "2026-09-05T10:02:00-03:00", {"reward": 0.7})
    status = audit.status()
    assert status["chain_valid"] is True
    assert status["events"] == 4
    audit.close()
