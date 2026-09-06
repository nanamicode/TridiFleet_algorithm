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


def test_fast_status_does_not_require_full_chain_scan():
    audit = AuditLog(":memory:")
    audit.start_run("2026-09-05T10:00:00-03:00", {"seed": 1})
    for i in range(250):
        audit.append("tick", f"2026-09-05T10:{i%60:02d}:00-03:00", {"i": i})
    fast = audit.status()
    full = audit.full_verify_status()
    assert fast["events"] == full["events"] == 251
    assert full["chain_valid"] is True
    audit.close()


def test_full_verify_detects_tampering():
    audit = AuditLog(":memory:")
    audit.start_run("2026-09-05T10:00:00-03:00", {"seed": 9})
    audit.append("decision", "2026-09-05T10:01:00-03:00", {"ad": "A"})
    audit.flush()
    audit.conn.execute(
        "UPDATE events SET payload_json=? WHERE kind='decision'",
        ('{"ad":"B"}',),
    )
    audit.conn.commit()
    status = audit.full_verify_status()
    assert status["chain_valid"] is False
    audit.close()
